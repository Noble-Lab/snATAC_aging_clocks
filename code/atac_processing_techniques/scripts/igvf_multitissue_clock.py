"""
PFC → IGVF multi-tissue ATAC age clock.

Loads per-tissue pseudobulk binary files built by build_igvf_pseudobulks.R,
then runs the same PFC → IGVF pipeline for each tissue and combined all-tissue.

All tissues share the same 883,711-peak reference; PFC × IGVF overlap =
439,522 shared features (same as SN-only run).

Outputs per tissue + combined:
  results/igvf_multitissue_results.csv
  figures/igvf_multitissue_violin_{cell type}.pdf/.png

External large-file dependencies (kept at fixed absolute paths, not inside
this repo):
  ~/data/parkinson_igvf/{pseudobulks,parkinson_atac_sn_86donors.h5ad} (~104GB raw)
  ~/data_back/PFC_brain_multiome/{final_atac_data,final_rna_data}.h5ad (14+18GB raw)
  ~/atac_processing_techniques/cache (this project's own cache/,
    holds several 100MB-1GB pseudobulk intermediates)
Everything else (results/, figures/, logs/) is local to this script's own
directory.
"""
import gc, pickle, sys, time, warnings
from collections import defaultdict
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test, log_coef

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, linregress, mannwhitneyu
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

ROOT      = Path(__file__).resolve().parent.parent
PB_DIR    = Path("~/data/parkinson_igvf/pseudobulks")
IGVF_H5AD = Path("~/data/parkinson_igvf/parkinson_atac_sn_86donors.h5ad")
CACHE     = Path("~/atac_processing_techniques/cache")  # large caches (100s of MB-1GB), kept out of the repo
PEAK_CSV  = CACHE / "igvf_sn_peak_coords.csv"
PFC_RNA   = "~/data_back/PFC_brain_multiome/final_rna_data.h5ad"
PFC_H5AD  = "~/data_back/PFC_brain_multiome/final_atac_data.h5ad"
RESULTS   = ROOT / "results"
FIG       = ROOT / "figures"

RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))
CHUNK_SIZE   = 5000
SEED         = 42
N_PEAKS_IGVF = 883711

# PFC CT modes available for training
PFC_CT_MAP = {"ExN": "Excitatory", "InN": "Inhibitory",
              "Oligo": "Oligo", "Astro": "Astro", "MG": "Microglia", "OPC": "OPC"}
# CT modes to evaluate per tissue (only those with PFC training data)
PFC_TRAINABLE = ["All-cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]

TISSUES = ["CBL", "CING", "MTG", "PUT", "SN"]

PD_PALETTE = {"yes": "#F44336", "no": "#2196F3"}
PD_ORDER   = ["no", "yes"]
PD_LABELS  = {"yes": "PD", "no": "Control"}


# ── normalisation ─────────────────────────────────────────────────────────────
def normalize(X):
    X = X.astype(np.float64)
    d = X.sum(1, keepdims=True); d = np.where(d < 1, 1, d)
    Z = np.log1p(X / d * 1e6)
    mu = Z.mean(1, keepdims=True); sd = Z.std(1, keepdims=True) + 1e-6
    return ((Z - mu) / sd).astype(np.float32)


# ── peak overlap (same as igvf_sn_clock.py) ──────────────────────────────────
def parse_peaks(peak_names):
    s = pd.Series(peak_names)
    colon = s.str.split(":", n=1, expand=True)
    chroms = colon[0].values
    se = colon[1].str.split("-", n=1, expand=True)
    return chroms, se[0].astype(np.int32).values, se[1].astype(np.int32).values


def compute_overlap_matrix(pa, pb):
    ca, sa, ea = parse_peaks(pa)
    cb, sb, eb = parse_peaks(pb)
    rows, cols = [], []
    for chrom in np.unique(np.concatenate([ca, cb])):
        ai = np.where(ca == chrom)[0]
        bi = np.where(cb == chrom)[0]
        if len(ai) == 0 or len(bi) == 0: continue
        ob = np.argsort(sb[bi]); bi_s = bi[ob]; sb_s = sb[bi][ob]; eb_s = eb[bi][ob]
        for i in ai:
            s_, e_ = sa[i], ea[i]
            r = np.searchsorted(sb_s, e_, side="left")
            if r == 0: continue
            msk = eb_s[:r] > s_
            if not msk.any(): continue
            for j in bi_s[:r][msk]:
                rows.append(i); cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(pa), len(pb)))


# ── helpers ───────────────────────────────────────────────────────────────────
def pearson_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[m], b[m])[0]) if m.sum() >= 3 else np.nan

def age_correction(actual, pred):
    ok = np.isfinite(actual) & np.isfinite(pred)
    slope, intercept, *_ = linregress(actual[ok], pred[ok])
    return pred - (slope * actual + intercept)

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / (s + 1e-9)


# ── load per-tissue pseudobulks from binary files ─────────────────────────────
def load_tissue_pseudobulks(tissue, keep_igvf):
    """
    Reads {tissue}_pb.bin + {tissue}_pb_meta.csv from PB_DIR.
    Returns dict: (donor_id, ct) → (counts_shared, age, pd_status)
    """
    bin_path  = PB_DIR / f"{tissue}_pb.bin"
    meta_path = PB_DIR / f"{tissue}_pb_meta.csv"

    if not bin_path.exists():
        print(f"  [WARN] {bin_path} not found — skipping {tissue}")
        return None, []

    meta = pd.read_csv(meta_path)
    n_rows = len(meta)
    raw = np.frombuffer(open(bin_path, "rb").read(), dtype=np.float32)
    mat = raw.reshape(n_rows, N_PEAKS_IGVF)   # R wrote column-major via t(mat)

    result = {}
    donors = []
    for _, row in meta.iterrows():
        donor_id  = row["donor_id"]
        ct        = row["ct"]
        age       = float(row["age"]) if pd.notna(row["age"]) else np.nan
        pd_status = str(row["pd_status"]) if pd.notna(row["pd_status"]) else "unknown"
        idx       = int(row["row_idx"])
        counts    = mat[idx, keep_igvf].astype(np.float32)
        result[(donor_id, ct)] = (counts, age, pd_status)
        if donor_id not in donors:
            donors.append(donor_id)

    print(f"  {tissue}: {len(result)} (donor,ct) pseudobulks, {len(donors)} donors")
    return result, donors


# ── build PFC per-CT pseudobulks ───────────────────────────────────────────────
def build_pfc_ct_pseudobulks(M_sub, pfc_donors, pfc_age):
    a_rna = ad.read_h5ad(PFC_RNA, backed="r")
    sid_age = a_rna.obs[["SampleID","Age"]].groupby("SampleID")["Age"].first().to_dict()
    a_rna.file.close()

    a   = ad.read_h5ad(PFC_H5AD, backed="r")
    obs = a.obs[["sample_id","cell_type"]].copy()
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["broad_ct"]  = obs["cell_type"].map(PFC_CT_MAP)
    obs["age"]       = obs["sample_id"].map(sid_age).astype(float)
    valid   = obs["age"].notna() & obs["broad_ct"].notna()
    all_bcs = a.obs_names.tolist()
    bc2meta = {bc: (r["sample_id"], r["broad_ct"])
               for bc, r in obs[valid].iterrows()}

    n_pfc = a.n_vars
    sums  = defaultdict(lambda: np.zeros(n_pfc, np.float32))

    t0 = time.time()
    for start in range(0, len(all_bcs), CHUNK_SIZE):
        end  = min(start + CHUNK_SIZE, len(all_bcs))
        vids = [bc for bc in all_bcs[start:end] if bc in bc2meta]
        if not vids: continue
        X = a[vids].to_memory().X
        if sp.issparse(X): X = sp.csr_matrix(X)
        X = X.astype(np.float32)
        idx_map = defaultdict(list)
        for ri, bc in enumerate(vids):
            idx_map[bc2meta[bc]].append(ri)
        for (donor, ct), idxs in idx_map.items():
            sums[(donor, ct)] += np.asarray(X[idxs].sum(0)).ravel()
        if (end // (CHUNK_SIZE * 10)) * (CHUNK_SIZE * 10) == end:
            print(f"    [pfc scan] {end}/{len(all_bcs)}  {(time.time()-t0)/60:.1f}m")
    a.file.close()
    print(f"    [pfc scan] done  {(time.time()-t0)/60:.1f}m")

    age_map = {d: pfc_age[i] for i, d in enumerate(pfc_donors)}
    ct_data = defaultdict(lambda: {"X": [], "ages": []})
    for (donor, ct), raw_sum in sums.items():
        if donor not in age_map: continue
        proj = np.asarray(raw_sum @ M_sub).ravel()
        ct_data[ct]["X"].append(proj.astype(np.float32))
        ct_data[ct]["ages"].append(float(age_map[donor]))
    del sums; gc.collect()
    return {ct: (np.vstack(d["X"]), np.array(d["ages"], np.float32))
            for ct, d in ct_data.items()}


# ── run predictions for one tissue ───────────────────────────────────────────
def run_tissue_predictions(pfc_pb_sel, pfc_age, pfc_pb_ct, tissue_pbs, tissue_donors, tissue_label="unknown"):
    preds = {}
    for ct_mode in PFC_TRAINABLE:
        if ct_mode == "All-cells":
            Xtr_raw, ytr = pfc_pb_sel.astype(np.float64), pfc_age
        else:
            if ct_mode not in pfc_pb_ct: continue
            Xtr_raw, ytr = pfc_pb_ct[ct_mode]
            Xtr_raw = Xtr_raw.astype(np.float64)
        Xtr = normalize(Xtr_raw)
        ts_data = [(d, tissue_pbs[(d, ct_mode)]) for d in tissue_donors
                   if (d, ct_mode) in tissue_pbs]
        if len(ts_data) < 3: continue
        Xts  = np.vstack([normalize(v[0][None,:].astype(np.float64)) for _,v in ts_data])
        m    = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, ytr)
        pred = m.predict(Xts)
        for rank, i in enumerate(np.argsort(np.abs(m.coef_))[::-1][:200]):
            log_coef("atac_processing_techniques", clock_name=f"igvf_pfc_train_{tissue_label}",
                     feature=f"feat_{i}", coefficient=float(m.coef_[i]),
                     modality="ATAC", cell_type=ct_mode, rank=rank, model_type="RidgeCV")
        actual  = np.array([v[1] for _,v in ts_data], np.float32)
        pd_col  = np.array([v[2] for _,v in ts_data])
        valid   = np.isfinite(actual)
        if valid.sum() < 3: continue
        preds[ct_mode] = (pred[valid], actual[valid], pd_col[valid])
        r   = pearson_r(actual[valid], pred[valid])
        mae = float(np.mean(np.abs(actual[valid] - pred[valid])))
        print(f"    {ct_mode:<12}: r={r:.3f}  MAE={mae:.1f}  "
              f"n_pd={(pd_col[valid]=='yes').sum()}  n_ctrl={(pd_col[valid]=='no').sum()}")
    return preds


# ── scatter + violin plots (multi-tissue grid) ────────────────────────────────
def make_multitissue_violin(all_preds, ct_mode, outpath_stem):
    tissues = [t for t in TISSUES + ["All-tissues"] if ct_mode in all_preds.get(t, {})]
    if not tissues: return
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(tissues), figsize=(2.5*len(tissues), 3.8))
    if len(tissues) == 1: axes = [axes]
    for ax, tissue in zip(axes, tissues):
        pred, actual, pd_col = all_preds[tissue][ct_mode]
        resid  = age_correction(actual, pred)
        r_pd   = resid[pd_col == "yes"]
        r_ctrl = resid[pd_col == "no"]
        if len(r_pd) < 2 or len(r_ctrl) < 2:
            ax.set_visible(False); continue
        parts = ax.violinplot([r_ctrl, r_pd], positions=[0,1],
                              widths=0.65, showmedians=True, showextrema=False)
        for pc, lb in zip(parts["bodies"], ["no","yes"]):
            pc.set_facecolor(PD_PALETTE[lb]); pc.set_alpha(0.7); pc.set_edgecolor("none")
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)
        for g, lb, pos in [(r_ctrl,"no",0),(r_pd,"yes",1)]:
            ax.scatter(pos + rng.uniform(-0.14,0.14,len(g)), g, s=10,
                       color=PD_PALETTE[lb], alpha=0.65, zorder=3, edgecolors="none")
        ax.axhline(0, color="black", lw=0.5, ls="--")
        if len(r_ctrl) >= 3 and len(r_pd) >= 3:
            _, p = mannwhitneyu(r_pd, r_ctrl, alternative="two-sided")
            d    = cohens_d(r_pd, r_ctrl)
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            annot = f"n={len(r_ctrl)}/{len(r_pd)}\nd={d:.2f}\np={p:.3f}{'' if stars=='ns' else stars}"
            ax.text(0.97,0.97, annot, transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Control","PD"], fontsize=6, rotation=15)
        ax.tick_params(labelsize=6)
        ax.set_title(tissue, fontsize=8, fontweight="bold")
    axes[0].set_ylabel("Age-corr. residual", fontsize=7)
    leg = [Line2D([0],[0], marker="o", color="w",
                  markerfacecolor=PD_PALETTE[v], markersize=7, label=PD_LABELS[v])
           for v in PD_ORDER]
    fig.legend(handles=leg, title="Diagnosis", title_fontsize=7, fontsize=7,
               frameon=False, bbox_to_anchor=(1.0,0.92), loc="upper left")
    fig.suptitle(f"Age-corrected residuals: PD vs Control  [{ct_mode}]  (IGVF 5 tissues)",
                 fontsize=8, y=1.01)
    fig.tight_layout()
    p = Path(outpath_stem)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── PFC cache + overlap ───────────────────────────────────────────────────
    print("Loading PFC cache and computing overlap…")
    with open(CACHE / "pfc_peak_pseudobulk.pkl", "rb") as f:
        pfc_pb_raw = pickle.load(f)
    pfc_donors  = sorted(pfc_pb_raw["donors"])
    pfc_age_all = pfc_pb_raw["age"].astype(np.float32)

    peak_df = pd.read_csv(PEAK_CSV)
    igvf_coords = (peak_df["chrom"] + ":" +
                   peak_df["start"].astype(str) + "-" +
                   peak_df["end"].astype(str)).tolist()

    t0 = time.time()
    M = compute_overlap_matrix(pfc_pb_raw["peak_names"], igvf_coords)
    keep_igvf = np.where(np.asarray(M.sum(0)).ravel() > 0)[0]
    M_sub     = M[:, keep_igvf]
    print(f"  {len(keep_igvf):,} shared features  ({time.time()-t0:.1f}s)")
    del M; gc.collect()

    pfc_pb_shared = np.asarray(pfc_pb_raw["peak_sum"] @ M_sub, dtype=np.float32)

    # ── PFC per-CT (single scan for all tissues) ──────────────────────────────
    print("Building PFC per-CT pseudobulks (one scan)…")
    pfc_pb_ct = build_pfc_ct_pseudobulks(M_sub, pfc_donors, pfc_age_all)
    gc.collect()

    # ── Per-tissue predictions ────────────────────────────────────────────────
    all_preds = {}
    summary_rows = []

    for tissue in TISSUES:
        print(f"\n── {tissue} ──")
        sn_bin = PB_DIR / "SN_pb.bin"
        if tissue == "SN" and not sn_bin.exists():
            # Fallback: scan SN h5ad directly if R binary not ready
            pbs, donors = _load_sn_pseudobulks_inline(keep_igvf)
        else:
            pbs, donors = load_tissue_pseudobulks(tissue, keep_igvf)

        if pbs is None or len(donors) == 0:
            print(f"  No data for {tissue}, skipping")
            continue

        preds = run_tissue_predictions(pfc_pb_shared, pfc_age_all,
                                       pfc_pb_ct, pbs, donors, tissue_label=tissue)
        all_preds[tissue] = preds

        for ct, (pred, actual, pd_col) in preds.items():
            r_pd   = age_correction(actual, pred)[pd_col == "yes"]
            r_ctrl = age_correction(actual, pred)[pd_col == "no"]
            pr  = pearson_r(actual, pred)
            mae = float(np.mean(np.abs(actual - pred)))
            d_val = p_val = np.nan
            if len(r_ctrl) >= 3 and len(r_pd) >= 3:
                _, p_val = mannwhitneyu(r_pd, r_ctrl, alternative="two-sided")
                d_val    = cohens_d(r_pd, r_ctrl)
            summary_rows.append(dict(tissue=tissue, ct=ct, pearson_r=pr, mae=mae,
                                     cohens_d=d_val, mwu_p=p_val,
                                     n_pd=int((pd_col=='yes').sum()),
                                     n_ctrl=int((pd_col=='no').sum())))

    # ── All-tissue combined pseudobulk ────────────────────────────────────────
    print("\n── All-tissue combined ──")
    all_tissue_pbs = {}
    donor_tissue_counts = defaultdict(int)

    for tissue in TISSUES:
        bin_path  = PB_DIR / f"{tissue}_pb.bin"
        meta_path = PB_DIR / f"{tissue}_pb_meta.csv"
        if tissue == "SN":
            # reload from binary if available, else skip
            if not bin_path.exists():
                print("  SN binary not available — skipping from combined")
                continue
        if not bin_path.exists(): continue

        meta = pd.read_csv(meta_path)
        raw  = np.frombuffer(open(bin_path, "rb").read(), dtype=np.float32)
        mat  = raw.reshape(len(meta), N_PEAKS_IGVF)

        for _, row in meta.iterrows():
            donor_id  = row["donor_id"]
            ct        = row["ct"]
            age       = float(row["age"]) if pd.notna(row["age"]) else np.nan
            pd_status = str(row["pd_status"]) if pd.notna(row["pd_status"]) else "unknown"
            idx       = int(row["row_idx"])
            counts    = mat[idx, keep_igvf].astype(np.float32)

            key = (donor_id, ct)
            if key not in all_tissue_pbs:
                all_tissue_pbs[key] = [counts, age, pd_status]
            else:
                all_tissue_pbs[key][0] = all_tissue_pbs[key][0] + counts

    all_tissue_donors = sorted(set(k[0] for k in all_tissue_pbs))
    # Convert lists to tuples
    all_tissue_pbs_final = {k: (v[0], v[1], v[2]) for k, v in all_tissue_pbs.items()}

    print(f"  Combined: {len(all_tissue_pbs_final)} (donor,ct) entries, {len(all_tissue_donors)} donors")
    preds_combined = run_tissue_predictions(pfc_pb_shared, pfc_age_all,
                                            pfc_pb_ct, all_tissue_pbs_final, all_tissue_donors,
                                            tissue_label="All-tissues")
    all_preds["All-tissues"] = preds_combined

    for ct, (pred, actual, pd_col) in preds_combined.items():
        r_pd   = age_correction(actual, pred)[pd_col == "yes"]
        r_ctrl = age_correction(actual, pred)[pd_col == "no"]
        pr  = pearson_r(actual, pred)
        mae = float(np.mean(np.abs(actual - pred)))
        d_val = p_val = np.nan
        if len(r_ctrl) >= 3 and len(r_pd) >= 3:
            _, p_val = mannwhitneyu(r_pd, r_ctrl, alternative="two-sided")
            d_val    = cohens_d(r_pd, r_ctrl)
        summary_rows.append(dict(tissue="All-tissues", ct=ct, pearson_r=pr, mae=mae,
                                 cohens_d=d_val, mwu_p=p_val,
                                 n_pd=int((pd_col=='yes').sum()),
                                 n_ctrl=int((pd_col=='no').sum())))

    # ── Save summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS / "igvf_multitissue_results.csv", index=False)
    print("\n── Summary ──")
    print(df.to_string(index=False))

    for _, r in df.iterrows():
        log_test("atac_processing_techniques",
                 f"igvf_multitissue_violin_{str(r['ct']).replace('-', '_')}",
                 analysis=f"PFC-trained Ridge clock -> IGVF {r['tissue']} (PD vs control)",
                 test="mannwhitneyu_age_residual_pd_vs_ctrl", group_a="PD", group_b="control",
                 n_a=r.get("n_pd"), n_b=r.get("n_ctrl"), statistic=r.get("pearson_r"),
                 effect_size=r.get("cohens_d"), p_value=r.get("mwu_p"),
                 notes=f"tissue={r['tissue']}; MAE={r.get('mae')}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    for ct_mode in PFC_TRAINABLE:
        has = [t for t in TISSUES + ["All-tissues"] if ct_mode in all_preds.get(t, {})]
        if not has: continue
        make_multitissue_violin(all_preds, ct_mode,
                                FIG / f"igvf_multitissue_violin_{ct_mode.replace('-','_')}")

    with open(RESULTS / "igvf_multitissue_predictions.pkl", "wb") as f:
        pickle.dump(all_preds, f)
    print("Saved igvf_multitissue_predictions.pkl")


def _load_sn_pseudobulks_inline(keep_igvf):
    """Load SN pseudobulks directly from h5ad (same logic as igvf_sn_clock.py)."""
    from collections import defaultdict
    IGVF_CT_MAP = {
        "oligodendrocyte": "Oligo",
        "astrocyte": "Astro",
        "microglial cell": "Microglia",
        "oligodendrocyte precursor cell": "OPC",
    }
    a   = ad.read_h5ad(IGVF_H5AD, backed="r")
    obs = a.obs[["DonorID","CL_term_name","age_at_death","PD"]].copy()
    all_bcs  = a.obs_names.tolist()
    n_igvf   = a.n_vars
    n_shared = len(keep_igvf)

    donor_meta = (obs.drop_duplicates("DonorID")
                  .set_index("DonorID")[["age_at_death","PD"]].to_dict("index"))

    sums_all = defaultdict(lambda: np.zeros(n_igvf, np.float64))
    sums_ct  = defaultdict(lambda: np.zeros(n_igvf, np.float64))

    t0 = time.time()
    for start in range(0, len(all_bcs), CHUNK_SIZE):
        end  = min(start + CHUNK_SIZE, len(all_bcs))
        vids = all_bcs[start:end]
        X    = a[vids].to_memory().X
        if sp.issparse(X): X = X.tocsr()
        X = X.astype(np.float32)
        chunk_obs = obs.iloc[start:end]
        idx_map_all = defaultdict(list)
        idx_map_ct  = defaultdict(list)
        for ri, (_, row) in enumerate(chunk_obs.iterrows()):
            donor = row["DonorID"]
            ct    = IGVF_CT_MAP.get(row["CL_term_name"])
            idx_map_all[donor].append(ri)
            if ct is not None:
                idx_map_ct[(donor, ct)].append(ri)
        for donor, idxs in idx_map_all.items():
            sums_all[donor] += np.asarray(X[idxs].sum(0)).ravel()
        for (donor, ct), idxs in idx_map_ct.items():
            sums_ct[(donor, ct)] += np.asarray(X[idxs].sum(0)).ravel()
        if (end // (CHUNK_SIZE * 10)) * (CHUNK_SIZE * 10) == end:
            print(f"    [sn scan] {end}/{len(all_bcs)}  {(time.time()-t0)/60:.1f}m")
    a.file.close()
    print(f"    [sn scan] done  {(time.time()-t0)/60:.1f}m")

    result = {}
    for donor, counts in sums_all.items():
        meta = donor_meta[donor]
        result[(donor, "All-cells")] = (counts[keep_igvf].astype(np.float32),
                                         float(meta["age_at_death"]), meta["PD"])
    for (donor, ct), counts in sums_ct.items():
        meta = donor_meta[donor]
        result[(donor, ct)] = (counts[keep_igvf].astype(np.float32),
                                float(meta["age_at_death"]), meta["PD"])
    donors = sorted(donor_meta.keys())
    print(f"    SN: {len(result)} (donor,ct) pseudobulks, {len(donors)} donors")
    return result, donors


if __name__ == "__main__":
    main()
