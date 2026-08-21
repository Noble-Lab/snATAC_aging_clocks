"""
PFC all-cell ATAC clock -> each retinal-native cell type (mouse, GSE325478 multiome).

Trains one Ridge clock on all 357 PFC donors pooled together (no cell-type
stratification), then predicts it separately onto each retinal-native cell
type's pseudobulk, asking whether the coarse, cell-type-agnostic PFC clock
still tracks age within each individual retinal cell type's chromatin.

Builds cache/multiome_pseudobulks.pkl if not already cached: per-sample,
per-retinal-cell-type pseudobulk ATAC profiles from the GSE325478 mouse
retinal multiome dataset, projected onto the same mm10 50kb-tile feature
space as the PFC clock (feat_idx_tile50kb.pkl). Requires
multiome_celltypes.py to have been run first (cache/multiome_cell_obs.csv).

PFC ATAC peaks are restricted to 50kb tiles with complete nonzero coverage
across all retinal samples and all PFC donors, to avoid a per-sample
peak-calling-completeness confound.

Saves: figures/multiome/analysis_allcell_clock_per_ct_scatter.pdf
       results/multiome_allcell_clock_per_ct.csv

External large-file dependencies (kept at fixed absolute paths, not inside
this repo): raw multiome_raw/*.h5 (2.0GB), this project's cache/ dir
(~/pfc_to_retinal/cache, several 100MB-2.7GB intermediates),
~/pfc_to_mouse/cache/{feat_idx_tile50kb,pfc_X_tile50kb}.pkl,
and ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
(238MB). Everything else (figures/, results/, logs/) is local to this
script's own directory.
"""
import gc
import logging
import pickle
import re
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test, log_coef

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
np.random.seed(42)

ROOT    = Path(__file__).resolve().parent
CACHE   = Path("~/pfc_to_retinal/cache")  # large caches (100s of MB-GB), kept out of the repo
FIG     = ROOT / "figures" / "multiome"
RESULTS = ROOT / "results"
LOG     = ROOT / "logs"

RAW_DIR      = Path("~/pfc_to_retinal/multiome_raw")  # raw GEO data (2.0GB), kept out of the repo
FEAT_IDX_PKL = Path("~/pfc_to_mouse/cache/feat_idx_tile50kb.pkl")
PFC_X_PKL    = Path("~/pfc_to_mouse/cache/pfc_X_tile50kb.pkl")
PFC_META_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))
MIN_SAMPLES  = 3
MIN_CELLS    = 10
TILE_SIZE    = 50_000

RET_CT_ORDER = ["Rod", "Cone", "Bipolar", "RGC", "Amacrine", "Horizontal",
                "Muller", "Astrocyte", "Microglia", "Endothelial", "RPE", "Pericyte"]

MM10_CHROM_SIZES = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680,
    "chr4": 156508116, "chr5": 151834684, "chr6": 149736546,
    "chr7": 145441459, "chr8": 129401213, "chr9": 124595110,
    "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685,
    "chr16": 98207768,  "chr17": 94987271,  "chr18": 90702639,
    "chr19": 61431566,  "chrX": 171031299,  "chrY": 91744698,
}

for d in (CACHE, FIG, RESULTS, LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG / "multiome_allcell_clock_per_ct.log", mode="w"),
              logging.StreamHandler()],
    force=True,
)
log = logging.getLogger()


# ── Cache build: per-sample pseudobulks, all-cell + per retinal cell type ──
def build_sample_meta() -> dict:
    files = sorted(RAW_DIR.glob("*.h5"))
    meta = {}
    pat = re.compile(r"(GSM\d+)_(IP(\d+)wk\d*)_multiome")
    for f in files:
        m = pat.search(f.name)
        if not m:
            continue
        gsm, label, wk = m.group(1), m.group(2), int(m.group(3))
        meta[gsm] = {"path": f, "label": label, "age_wk": wk}
    log.info(f"Found {len(meta)} multiome samples")
    return meta


def _build_tile_tables():
    offsets, n_tiles = {}, {}
    idx = 0
    for ch, sz in sorted(MM10_CHROM_SIZES.items()):
        offsets[ch] = idx
        nt = (sz + TILE_SIZE - 1) // TILE_SIZE
        n_tiles[ch] = nt
        idx += nt
    return offsets, n_tiles, idx

CHR_OFFSET, CHR_N_TILES, N_TILES_TOTAL = _build_tile_tables()


def make_feat_to_col(feat_idx: np.ndarray) -> np.ndarray:
    lut = np.full(N_TILES_TOTAL, -1, dtype=np.int32)
    for ci, gi in enumerate(feat_idx):
        lut[int(gi)] = ci
    return lut


def peaks_to_tile_projection(peak_names: list, feat_to_col: np.ndarray,
                              n_feat: int) -> sp.csr_matrix:
    """Peak (mm10 native, per-sample) -> shared 50kb tile column projection."""
    s = pd.Series(peak_names)
    parts = s.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    chrom = parts["chrom"].values
    start = parts["start"].fillna(-1).values.astype(np.int64)

    n_peaks = len(peak_names)
    col = np.full(n_peaks, -1, dtype=np.int64)
    for ch in np.unique(chrom):
        if ch not in CHR_OFFSET:
            continue
        mask = chrom == ch
        n_t = CHR_N_TILES[ch]
        tile_local = np.clip(start[mask] // TILE_SIZE, 0, n_t - 1)
        gi = CHR_OFFSET[ch] + tile_local
        col[mask] = feat_to_col[gi]

    valid = col >= 0
    rows = np.where(valid)[0]
    P = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, col[valid])),
                       shape=(n_peaks, n_feat))
    return P


def build_pseudobulks(sample_meta: dict, obs_df: pd.DataFrame,
                       feat_to_col: np.ndarray, n_feat: int) -> dict:
    cp = CACHE / "multiome_pseudobulks.pkl"
    if cp.exists():
        log.info(f"Loading cached pseudobulks from {cp} …")
        with open(cp, "rb") as f:
            return pickle.load(f)

    ret_cts = sorted(obs_df["cell_type"].unique())
    log.info(f"Retinal-native CTs ({len(ret_cts)}): {ret_cts}")

    all_cell_records = []
    ct_pb = {ct: {} for ct in ret_cts}

    obs_df = obs_df.copy()
    obs_df["gsm"] = obs_df["gsm"].astype(str)
    obs_df["barcode"] = obs_df["barcode"].astype(str)

    for gsm in sorted(sample_meta.keys()):
        m = sample_meta[gsm]
        log.info(f"  Pseudobulking {gsm} ({m['label']}, {m['age_wk']}wk) …")
        a = sc.read_10x_h5(str(m["path"]), gex_only=False)
        a.var_names_make_unique()
        atac = a[:, a.var["feature_types"] == "Peaks"].copy()
        del a
        gc.collect()

        peak_names = atac.var_names.tolist()
        barcodes = atac.obs_names.tolist()
        X = atac.X.tocsr().astype(np.float32)
        del atac
        gc.collect()

        P = peaks_to_tile_projection(peak_names, feat_to_col, n_feat)
        cell_tile = (X @ P).tocsr()
        del X, P
        gc.collect()

        pb_all = np.asarray(cell_tile.sum(axis=0)).ravel()
        all_cell_records.append({"gsm": gsm, "age_wk": m["age_wk"], "label": m["label"],
                                  "n_cells": len(barcodes), "pb": pb_all})

        gsm_obs = obs_df[obs_df["gsm"] == gsm].set_index("barcode")
        bc_to_row = {bc: i for i, bc in enumerate(barcodes)}

        for ct in ret_cts:
            ct_bcs = gsm_obs.index[gsm_obs["cell_type"] == ct]
            rows = [bc_to_row[bc] for bc in ct_bcs if bc in bc_to_row]
            if len(rows) < MIN_CELLS:
                continue
            pb = np.asarray(cell_tile[rows].sum(axis=0)).ravel()
            ct_pb[ct][gsm] = {"pb": pb, "n_cells": len(rows),
                               "age_wk": m["age_wk"], "label": m["label"]}

        log.info(f"    {len(barcodes)} cells → all-cell pb + "
                 f"{sum(1 for ct in ret_cts if gsm in ct_pb[ct])} retinal-CT pbs")
        del cell_tile
        gc.collect()

    result = {"all_cell": all_cell_records, "ct_pb": ct_pb}
    with open(cp, "wb") as f:
        pickle.dump(result, f)
    log.info("Saved pseudobulks")
    return result


def load_pseudobulks() -> dict:
    obs_path = CACHE / "multiome_cell_obs.csv"
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} not found — run multiome_celltypes.py first")
    obs_df = pd.read_csv(obs_path, dtype={"barcode": str, "gsm": str})
    log.info(f"Loaded cell obs: {len(obs_df)} cells")

    with open(FEAT_IDX_PKL, "rb") as f:
        feat_idx = np.asarray(pickle.load(f))
    feat_to_col = make_feat_to_col(feat_idx)
    n_feat = len(feat_idx)
    log.info(f"Feature space: {n_feat} mm10 50kb tiles")

    sample_meta = build_sample_meta()
    return build_pseudobulks(sample_meta, obs_df, feat_to_col, n_feat)


# ── Clock training + plotting ───────────────────────────────────────────────
def cpm_zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    X = np.log1p(X / rs * 1e6)
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def ct_pb_to_df(pb_dict_for_ct: dict) -> pd.DataFrame:
    rows = []
    for gsm, d in pb_dict_for_ct.items():
        rows.append({"gsm": gsm, "age_wk": d["age_wk"], "n_cells": d["n_cells"], "pb": d["pb"]})
    return pd.DataFrame(rows)


def plot_scatter_grid(items: list, suptitle: str, filename: str, ncols: int = 4):
    n = len(items)
    ncols = min(ncols, max(n, 1))
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols + 0.7, 3.9 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    mappable = None
    for i, item in enumerate(items):
        ax = axes_flat[i]
        aw = np.asarray(item["age_wk"], dtype=float)
        pr = np.asarray(item["pred"], dtype=float)
        mappable = ax.scatter(aw, pr, c=aw, cmap="plasma", vmin=5, vmax=120, s=60,
                              zorder=3, edgecolors="k", linewidths=0.4)
        if len(aw) >= 2:
            mcoef, b = np.polyfit(aw, pr, 1)
            xs = np.linspace(aw.min(), aw.max(), 100)
            ax.plot(xs, mcoef * xs + b, "k--", lw=1.2, alpha=0.6, zorder=2)
        ax.set_title(f"{item['title']}\nr={item['r']:+.3f} p={item['p_r']:.4f}   "
                     f"ρ={item['rho']:+.3f} p={item['p_s']:.4f}  (n={item['n']})", fontsize=9)
        ax.set_xlabel("Age (weeks)", fontsize=8)
        ax.set_ylabel("Predicted age (yr)", fontsize=8)
        ax.tick_params(labelsize=7)
        if item.get("desc"):
            ax.text(0.05, 0.95, item["desc"], transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left", color="dimgray")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    if mappable is not None:
        cax = fig.add_axes([0.955, 0.15, 0.013, 0.7])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("Age (weeks)", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{filename}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {filename}")


def main():
    log.info("=" * 70)
    log.info("PFC ALL-CELL clock -> each mouse retinal-native cell type")
    log.info("=" * 70)

    with open(PFC_X_PKL, "rb") as f:
        pfc_X_all = np.asarray(pickle.load(f), dtype=np.float64)
    with open(PFC_META_PKL, "rb") as f:
        pfc_meta = pickle.load(f)
    pfc_age_all = np.asarray(pfc_meta["age"], dtype=np.float64)
    log.info(f"PFC all-cell training: {pfc_X_all.shape}, age {pfc_age_all.min():.0f}-{pfc_age_all.max():.0f}yr")

    pb = load_pseudobulks()

    ret_X = np.stack([r["pb"] for r in pb["all_cell"]])
    ret_nz = (ret_X > 0).sum(axis=0)
    pfc_nz = (pfc_X_all > 0).sum(axis=0)
    mask = (ret_nz == ret_X.shape[0]) & (pfc_nz == pfc_X_all.shape[0])
    log.info(f"Robust tile mask: {int(mask.sum())}/{len(mask)} tiles")

    pfc_X_masked = pfc_X_all[:, mask]
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge.fit(cpm_zscore(pfc_X_masked), pfc_age_all)
    log.info(f"PFC all-cell Ridge fit (alpha={ridge.alpha_:.4g}, n={len(pfc_age_all)})")
    mask_idx = np.where(mask)[0]
    for rank, i in enumerate(np.argsort(np.abs(ridge.coef_))[::-1][:200]):
        log_coef("pfc_to_retinal", clock_name="pfc_allcell_to_retinal_ridge_clock",
                 feature=f"tile_{mask_idx[i]}", coefficient=float(ridge.coef_[i]),
                 modality="ATAC", cell_type="All-cells", rank=rank, model_type="RidgeCV")

    ret_cts = [c for c in RET_CT_ORDER if c in pb["ct_pb"] and len(pb["ct_pb"][c]) >= MIN_SAMPLES]
    log.info(f"Retinal cell types with pseudobulks (n>={MIN_SAMPLES}): {ret_cts}")

    items, records = [], []
    for ret_ct in ret_cts:
        df = ct_pb_to_df(pb["ct_pb"][ret_ct])
        Xr = np.stack(df["pb"].values)[:, mask]
        age_wk = df["age_wk"].values
        n_cells = df["n_cells"].values
        preds = ridge.predict(cpm_zscore(Xr))
        r, p_r = pearsonr(age_wk, preds)
        rho, p_s = spearmanr(age_wk, preds)
        log.info(f"  All-cells → {ret_ct:<12} r={r:+.3f} p={p_r:.4f}  rho={rho:+.3f} p={p_s:.4f}  "
                 f"n={len(age_wk)}  cells/sample {n_cells.min()}-{n_cells.max()}")
        log_test("pfc_to_retinal", "analysis_allcell_clock_per_ct_scatter",
                 analysis="PFC all-cell -> mouse retinal cell-type Ridge clock",
                 test="pearson_correlation", group_a="All-cells", group_b=ret_ct,
                 n_a=len(age_wk), statistic=r, p_value=p_r)
        log_test("pfc_to_retinal", "analysis_allcell_clock_per_ct_scatter",
                 analysis="PFC all-cell -> mouse retinal cell-type Ridge clock",
                 test="spearman_correlation", group_a="All-cells", group_b=ret_ct,
                 n_a=len(age_wk), statistic=rho, p_value=p_s)
        items.append({"title": f"PFC All-cells → Retinal {ret_ct}", "age_wk": age_wk, "pred": preds,
                      "r": r, "p_r": p_r, "rho": rho, "p_s": p_s, "n": len(age_wk),
                      "desc": f"cells/sample: {n_cells.min()}-{n_cells.max()}"})
        records.append({"pfc_ct": "All-cells", "ret_ct": ret_ct, "r": r, "p_r": p_r,
                        "rho": rho, "p_s": p_s, "n": len(age_wk)})

    plot_scatter_grid(items,
                       "PFC ALL-CELL clock → each retinal-native cell type (mouse GSE325478)",
                       "analysis_allcell_clock_per_ct_scatter", ncols=4)

    df_out = pd.DataFrame(records)
    df_out.to_csv(RESULTS / "multiome_allcell_clock_per_ct.csv", index=False)
    log.info("\n" + df_out.to_string(index=False))
    log.info("\nDone.")


if __name__ == "__main__":
    main()
