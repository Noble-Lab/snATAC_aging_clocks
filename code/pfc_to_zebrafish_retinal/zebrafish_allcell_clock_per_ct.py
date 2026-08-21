"""
PFC all-cell ATAC clock -> each marked zebrafish retinal cell type.

Trains one Ridge clock on all 357 PFC donors pooled together (no cell-type
stratification), then predicts it separately onto each zebrafish retinal
cell type's pseudobulk, asking whether the coarse, cell-type-agnostic PFC
clock still tracks age within each individual retinal cell type's profile.

PFC ATAC peaks are projected onto the shared danRer11 50kb tile grid used for
the retinal ATAC channel; tiles are further restricted to those with
near-complete coverage in both species (build_atac_coverage_mask) so that
per-sample peak-calling gaps don't masquerade as biological signal.

Requires zebrafish_multiome_celltypes.py to have been run first.
Saves: figures/multiome_clocks_final/scatter_allcellclock_percelltype_atac.pdf
       results/multiome_clocks_final/allcell_clock_per_ct.csv

Reuses zebrafish_multiome_celltypes.py's ROOT/CACHE (see that file's header
for its external large-file dependencies); figures/, results/, and logs/
here are local to this script's own directory.
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

import zebrafish_multiome_celltypes as ct1
from np_compat_io import safe_pickle_load

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CACHE   = ct1.CACHE
BC_DIR  = ct1.BC_DIR
FIG     = ct1.ROOT / "figures" / "multiome_clocks_final"
RESULTS = ct1.ROOT / "results" / "multiome_clocks_final"
LOG     = ct1.LOG
CT_COLORS = ct1.CT_COLORS
PFC_CTS   = ct1.PFC_CTS

OBS_CSV = CACHE / "retinal_multiome_obs.csv"
V3_PKL  = ct1.V3_PKL

# Cached retinal pseudobulks are shared with zebrafish_multiome_ct_clocks.py's
# per-cell-type clock and live under its results directory.
PB_CACHE_DIR = ct1.ROOT / "results" / "multiome_clocks"

RIDGE_ALPHAS     = tuple(np.logspace(-2, 6, 30))
MIN_CELLS_RETINA = 10  # min cells per (sample, cell type) to pseudobulk
MIN_CELLS_PFC    = 10  # min cells per (donor, cell type) to include in PFC training

for d in (FIG, RESULTS, LOG, PB_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG / "allcell_clock_per_ct.log", mode="w"),
              logging.StreamHandler()],
    force=True,
)
log = logging.getLogger()


def cpm_zscore(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    X = np.log1p(X / rs * 1e6)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


# ── Retinal pseudobulks (ATAC-tile) per cell type ─────────────────────────────
def build_retinal_pseudobulks(obs_df: pd.DataFrame, bucket_col: str, bucket_values: list) -> dict:
    cache_path = PB_CACHE_DIR / f"retinal_pb_atac_{bucket_col or 'allcells'}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    gsm_list = sorted(ct1.SAMPLE_META.keys())
    result = {b: {"rows": [], "age_mo": [], "gsm": [], "n_cells": []}
              for b in (bucket_values or ["all_cells"])}

    for gsm in gsm_list:
        mat = load_npz(str(BC_DIR / f"{gsm}_atac_tile.npz"))
        with open(BC_DIR / f"{gsm}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        bc_to_row = {bc: i for i, bc in enumerate(meta["barcodes"])}

        sub_obs = obs_df[obs_df["gsm"] == gsm].copy()
        sub_obs["row"] = sub_obs["barcode"].map(bc_to_row)
        sub_obs = sub_obs.dropna(subset=["row"])
        row_idx = sub_obs["row"].astype(int).values
        mat_qc = mat[row_idx]
        age_mo = ct1.SAMPLE_META[gsm][0]

        if bucket_col is None:
            n = mat_qc.shape[0]
            if n >= MIN_CELLS_RETINA:
                pb = np.asarray(mat_qc.sum(axis=0)).ravel()
                result["all_cells"]["rows"].append(pb)
                result["all_cells"]["age_mo"].append(age_mo)
                result["all_cells"]["gsm"].append(gsm)
                result["all_cells"]["n_cells"].append(n)
        else:
            labels = sub_obs[bucket_col].values
            for b in bucket_values:
                bmask = labels == b
                n = int(bmask.sum())
                if n < MIN_CELLS_RETINA:
                    continue
                pb = np.asarray(mat_qc[bmask.astype(bool)].sum(axis=0)).ravel()
                result[b]["rows"].append(pb)
                result[b]["age_mo"].append(age_mo)
                result[b]["gsm"].append(gsm)
                result[b]["n_cells"].append(n)

    out = {}
    for b, d in result.items():
        if len(d["rows"]) >= 3:
            out[b] = {"X_raw": np.array(d["rows"]), "age_mo": np.array(d["age_mo"]),
                       "gsm": d["gsm"], "n_cells": d["n_cells"]}
        else:
            log.warning(f"  atac/{b}: only {len(d['rows'])} samples with "
                        f">= {MIN_CELLS_RETINA} cells -> dropped")
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    log.info(f"Retinal atac pseudobulks ({bucket_col or 'all_cells'}): "
             f"{ {b: len(v['gsm']) for b, v in out.items()} }")
    return out


# ── PFC training data: ATAC (native peaks -> danRer11 tiles) ──────────────────
def build_pfc_atac_train(v3: dict) -> dict:
    with open(ct1.PFC_DANRER11_LIFTOVER_PKL, "rb") as f:
        lo_idx = pickle.load(f)
    pfc_cols, inverse, n_tiles = lo_idx["pfc_cols"], lo_idx["inverse"], len(lo_idx["unique_tiles"])
    with open(ct1.PFC_PEAK_ALLCELLS_PKL, "rb") as f:
        pfc_all = pickle.load(f)
    donors, age_all = pfc_all["donors"], np.asarray(pfc_all["age"], dtype=np.float64)
    counts_atac = v3["cell_counts_by_ct_atac"].reindex(donors).fillna(0)

    def project(X_native):
        X_sub = X_native[:, pfc_cols]
        X_dr = np.zeros((X_native.shape[0], n_tiles), dtype=np.float64)
        np.add.at(X_dr.T, inverse, X_sub.T)
        return X_dr

    out = {"all_cells": {"X_raw": project(np.asarray(pfc_all["peak_sum"], dtype=np.float64)), "age": age_all}}
    ct_native = safe_pickle_load(ct1.PFC_PEAK_BY_CT_NATIVE_PKL)
    for ct in PFC_CTS:
        valid = counts_atac[ct].values >= MIN_CELLS_PFC
        out[ct] = {"X_raw": project(np.asarray(ct_native[ct][valid], dtype=np.float64)), "age": age_all[valid]}
    del ct_native
    return out


# ── ATAC tile-coverage-completeness mask ───────────────────────────────────────
def build_atac_coverage_mask(pfc_allcells_X: np.ndarray, retinal_allcells_X: np.ndarray,
                              pfc_min_frac: float = 0.95, retinal_min_frac: float = 1.0) -> np.ndarray:
    """
    Cell Ranger ARC calls ATAC peaks independently per sample: a tile with
    zero signal often just means "no peak happened to be called here in this
    sample," not "closed chromatin." That presence/absence pattern can
    dominate the leading PC and produce a null clock. PFC uses a relaxed 95%
    (not literal 100%) threshold because donor sequencing depth is
    long-tailed -- a handful of low-depth donors would otherwise zero out the
    vast majority of tiles.
    """
    pfc_nz = (pfc_allcells_X > 0).mean(axis=0) >= pfc_min_frac
    ret_nz = (retinal_allcells_X > 0).mean(axis=0) >= retinal_min_frac
    mask = pfc_nz & ret_nz
    log.info(f"ATAC tile coverage mask: {int(mask.sum())}/{len(mask)} tiles retained "
             f"(PFC >= {pfc_min_frac*100:.0f}% of {pfc_allcells_X.shape[0]} donors nonzero, "
             f"retinal >= {retinal_min_frac*100:.0f}% of {retinal_allcells_X.shape[0]} samples nonzero)")
    return mask


def apply_atac_mask(d: dict, mask: np.ndarray) -> dict:
    return {ct: {**v, "X_raw": v["X_raw"][:, mask]} for ct, v in d.items()}


# ── Fit on PFC all-cells, predict onto one retinal cell type ─────────────────
def fit_and_predict(pfc_d: dict, ret_X_raw: np.ndarray, ret_age_mo: np.ndarray,
                     ret_gsm: list, ret_ncells: list, normalizer):
    Xn_pfc = normalizer(pfc_d["X_raw"])
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge.fit(Xn_pfc, pfc_d["age"])

    Xn_ret = normalizer(ret_X_raw)
    if Xn_ret.shape[1] != ridge.coef_.shape[0]:
        return None
    preds = ridge.predict(Xn_ret)
    n = len(preds)
    if n >= 3:
        r, p_r = pearsonr(ret_age_mo, preds)
        rho, p_s = spearmanr(ret_age_mo, preds)
    else:
        r = p_r = rho = p_s = np.nan
    return {"predictions": preds, "age_mo": ret_age_mo, "gsm": ret_gsm, "n_cells": ret_ncells,
            "pearson_r": r, "pearson_p": p_r, "spearman_rho": rho, "spearman_p": p_s, "n": n}


# ── Plot ───────────────────────────────────────────────────────────────────
def _title(res, pfc_ct, ret_ct):
    return (f"{pfc_ct} -> {ret_ct}\n"
            f"r={res['pearson_r']:+.2f} (p={res['pearson_p']:.3f})   "
            f"ρ={res['spearman_rho']:+.2f} (p={res['spearman_p']:.3f})   n={res['n']}")


def plot_scatter_grid(items, title, fname, get_color, ncols=4):
    """items: list of (res, pfc_ct, ret_ct). get_color(pfc_ct, ret_ct) -> hex color."""
    n = len(items)
    if n == 0:
        log.warning(f"No predictions for {fname} -> skipping"); return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    for i, (res, pfc_ct, ret_ct) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        col = get_color(pfc_ct, ret_ct)
        ax.scatter(res["age_mo"], res["predictions"], c=col, s=60, alpha=0.85,
                   zorder=3, edgecolors="k", linewidths=0.4)
        if res["n"] >= 2:
            m, b = np.polyfit(res["age_mo"], res["predictions"], 1)
            xs = np.linspace(min(res["age_mo"]), max(res["age_mo"]), 50)
            ax.plot(xs, m * xs + b, color=col, lw=1.5, alpha=0.6, ls="--")
        ax.set_title(_title(res, pfc_ct, ret_ct), fontsize=8.5)
        ax.set_xlabel("Actual age (months)", fontsize=8)
        ax.set_ylabel("Predicted age (yr)", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {fname}")


def main():
    log.info("=" * 70)
    log.info("PFC all-cell ATAC clock -> each marked zebrafish retinal cell type")
    log.info("=" * 70)

    obs_df = pd.read_csv(OBS_CSV)
    v3 = safe_pickle_load(V3_PKL)
    marker_cts = [c for c in CT_COLORS if c != "Unknown"]

    pfc_train = build_pfc_atac_train(v3)
    ret_all    = build_retinal_pseudobulks(obs_df, None, None)
    ret_marker = build_retinal_pseudobulks(obs_df, "cell_type", marker_cts)

    mask = build_atac_coverage_mask(pfc_train["all_cells"]["X_raw"], ret_all["all_cells"]["X_raw"])
    pfc_train  = apply_atac_mask(pfc_train, mask)
    ret_marker = apply_atac_mask(ret_marker, mask)

    d = pfc_train["all_cells"]
    predictions = {}
    for ret_ct in marker_cts:
        if ret_ct not in ret_marker:
            continue
        ret = ret_marker[ret_ct]
        res = fit_and_predict(d, ret["X_raw"], ret["age_mo"], ret["gsm"], ret["n_cells"], cpm_zscore)
        if res is None:
            continue
        predictions[ret_ct] = res
        log.info(f"  all_cells -> {ret_ct:<12} r={res['pearson_r']:+.3f} p={res['pearson_p']:.4f}  "
                 f"rho={res['spearman_rho']:+.3f} p={res['spearman_p']:.4f}  n={res['n']}")
        log_test("pfc_to_zebrafish_retinal", "scatter_allcellclock_percelltype_atac",
                 analysis="PFC all-cell -> zebrafish retinal cell-type Ridge clock (atac)",
                 test="pearson_correlation", group_a="all_cells", group_b=ret_ct,
                 n_a=res["n"], statistic=res["pearson_r"], p_value=res["pearson_p"])
        log_test("pfc_to_zebrafish_retinal", "scatter_allcellclock_percelltype_atac",
                 analysis="PFC all-cell -> zebrafish retinal cell-type Ridge clock (atac)",
                 test="spearman_correlation", group_a="all_cells", group_b=ret_ct,
                 n_a=res["n"], statistic=res["spearman_rho"], p_value=res["spearman_p"])

    items = [(res, "all_cells", ret_ct) for ret_ct, res in predictions.items()]
    plot_scatter_grid(items,
                       "PFC ALL-CELL clock -> each zebrafish retinal cell type (ATAC)",
                       "scatter_allcellclock_percelltype_atac",
                       get_color=lambda pc, rc: CT_COLORS.get(rc, "#555"))

    rows = [{"modality": "atac", "pfc_ct": "all_cells", "ret_ct": ret_ct,
             "pearson_r": res["pearson_r"], "pearson_p": res["pearson_p"],
             "spearman_rho": res["spearman_rho"], "spearman_p": res["spearman_p"],
             "n": res["n"]} for ret_ct, res in predictions.items()]
    df = pd.DataFrame(rows).sort_values("pearson_p")
    df.to_csv(RESULTS / "allcell_clock_per_ct.csv", index=False)
    log.info("\n" + df.to_string(index=False))
    log.info("\nDone.")


if __name__ == "__main__":
    main()
