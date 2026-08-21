"""
Train the PFC -> hippocampus ATAC Ridge clock (5kb tiles, CPM+log1p+per-donor
z-score) for all cells and each of 6 broad cell types, and plot predicted vs
actual hippocampus age as a single row of scatter panels.

Builds cache/hip_pseudobulks_nr2f2inh.pkl (hippocampus 5kb-tile pseudobulks,
all cells + per broad cell type) if not already
cached.

External large-file dependency (kept at a fixed absolute path, not inside
this repo): this project's cache/ dir, ~/pfc_to_hippocampus/cache,
also holds pfc_to_hip_atac_clock.py's and pfc_to_hip_atac_clock_peaks.py's
100s-of-MB caches, so it's kept at one stable location rather than split.
This script's own cache files are small (11MB/85MB).

Saves: figures/scatter_tile_zscore_horiz_NR2F2inh.pdf/.png
"""
import logging
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test, log_coef

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

ROOT  = Path(__file__).resolve().parent
CACHE = Path("~/pfc_to_hippocampus/cache")  # large caches (100s of MB), kept out of the repo
FIG   = ROOT / "figures"
LOG   = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG / "plot_scatter_tile_zscore_horiz_nr2f2inh.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

_PFC_CACHE_TMP = Path("/tmp/pfc_peak_pseudobulk.pkl")
_PFC_CACHE_NFS = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PFC_CACHE = _PFC_CACHE_TMP if _PFC_CACHE_TMP.exists() else _PFC_CACHE_NFS

HIP_ATAC = Path("~/data_back/GSE278576_hippocampus/processed/atac_tiles.h5ad")
HIP_META = Path("~/data_back/GSE278576_hippocampus/metadata.tsv.gz")

RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))
BROAD_CTS    = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
CHUNK_SIZE   = 10_000

# NR2F2 moved to Inhibitory (vs. Excitatory in the original HIP_CT_MAP)
HIP_CT_MAP = {
    "DG": "Excitatory",  "CA1": "Excitatory",  "CA2-CA3": "Excitatory",
    "SUB": "Excitatory",
    "NR2F2": "Inhibitory",
    "SST": "Inhibitory", "VIP": "Inhibitory",   "PVALB": "Inhibitory",
    "LAMP5": "Inhibitory", "Chandelier": "Inhibitory",
    "Oligo": "Oligo",   "Astro": "Astro",
    "Microglia": "Microglia", "Macro": "Microglia",
    "OPC": "OPC",
}


def build_hip_pseudobulks():
    """Return hippocampus 5kb-tile donor pseudobulks (all cells + per broad
    cell type), with NR2F2 classified as Inhibitory. Cached on disk."""
    cache_path = CACHE / "hip_pseudobulks_nr2f2inh.pkl"
    if cache_path.exists():
        log.info("Loading cached hippocampus pseudobulks (NR2F2->Inhibitory)")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log.info("Building hippocampus pseudobulks (NR2F2->Inhibitory) from atac_tiles.h5ad…")
    adata = ad.read_h5ad(HIP_ATAC, backed="r")
    tile_names = list(adata.var_names.astype(str))
    n_tiles = len(tile_names)

    meta_raw = pd.read_csv(HIP_META, sep="\t")
    meta_deep = meta_raw[meta_raw["bacrode"].str.contains("_deep_")].copy()
    meta_deep["bacrode"] = meta_deep["bacrode"].str.replace("_deep_", "_", regex=False)
    meta = pd.concat([meta_raw[~meta_raw["bacrode"].str.contains("_deep_")], meta_deep],
                     ignore_index=True).set_index("bacrode")

    obs = adata.obs.copy()
    obs["subclass"] = meta.reindex(obs.index)["subclass"].values
    obs["broad_ct"] = obs["subclass"].map(HIP_CT_MAP)

    donors_age = (obs[["donor_id", "age"]].drop_duplicates("donor_id")
                  .set_index("donor_id")["age"])
    donors = sorted(donors_age.index.tolist())
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    N = len(donors)
    age = donors_age.reindex(donors).values.astype(float)

    n_cts = len(BROAD_CTS)

    tile_sum_all = np.zeros((N, n_tiles), dtype=np.float64)
    cc_all       = np.zeros(N, dtype=np.int64)
    tile_sum_ct  = np.zeros((N, n_cts, n_tiles), dtype=np.float64)
    cc_ct        = np.zeros((N, n_cts), dtype=np.int64)

    dcodes_full  = obs["donor_id"].map(donor_to_idx).values.astype(np.intp)
    ct_vals_full = obs["broad_ct"].values

    cell_ids = adata.obs_names.values
    n_cells  = len(cell_ids)
    log.info(f"  Hippocampus: {N} donors, {n_cells:,} cells, {n_tiles} tiles")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        X = adata[ids].to_memory().X
        if sp.issparse(X):
            X = np.asarray(X.todense(), dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        dcodes = dcodes_full[start:end]
        ct_vals = ct_vals_full[start:end]

        for d in np.unique(dcodes):
            mask_d = dcodes == d
            tile_sum_all[d] += X[mask_d].sum(axis=0)
            cc_all[d]       += mask_d.sum()

        for cti, ct in enumerate(BROAD_CTS):
            mask_ct = (ct_vals == ct)
            if not mask_ct.any():
                continue
            for d in np.unique(dcodes[mask_ct]):
                mask_dc = mask_ct & (dcodes == d)
                tile_sum_ct[d, cti] += X[mask_dc].sum(axis=0)
                cc_ct[d, cti]       += mask_dc.sum()

        if (start // CHUNK_SIZE) % 10 == 0:
            log.info(f"    hip {end:,}/{n_cells:,}  ({time.time()-t0:.0f}s)")

    adata.file.close()
    log.info(f"  Done in {time.time()-t0:.0f}s")

    result = {
        "tile_names": tile_names,
        "donors": donors,
        "age": age,
        "all_cells": {"tile_sum": tile_sum_all, "cell_counts": cc_all},
        "per_ct": {
            ct: {"tile_sum": tile_sum_ct[:, i, :], "cell_counts": cc_ct[:, i]}
            for i, ct in enumerate(BROAD_CTS)
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache_path}")
    return result


def normalize_zscore(tile_sum, cell_counts):
    valid = cell_counts > 0
    X = tile_sum[valid].astype(np.float64)
    d = X.sum(axis=1, keepdims=True)
    d = np.where(d < 1.0, 1.0, d)
    X = np.log1p(X / d * 1e6)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd, valid


def run_ridge(X_train, y_train, X_test, y_test):
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, y_train)
    preds = m.predict(X_test)
    mask = np.isfinite(y_test) & np.isfinite(preds)
    if mask.sum() >= 3:
        r, p = pearsonr(y_test[mask], preds[mask])
        r, p = float(r), float(p)
    else:
        r, p = np.nan, np.nan
    mae = float(np.mean(np.abs(y_test[mask] - preds[mask]))) if mask.sum() >= 3 else np.nan
    return preds, y_test, r, mae, p, m


def scatter_panel(ax, y_true, y_pred, r, mae, title):
    color = "steelblue"
    ax.scatter(y_true, y_pred, s=18, alpha=0.7, edgecolors="none", color=color)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual age (yr)", fontsize=8)
    ax.set_ylabel("Predicted age (yr)", fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nMAE = {mae:.1f} yr\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def main():
    print("Loading caches…")
    with open(PFC_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    pfc_sum = pfc_all["peak_sum"].astype(np.float64)
    pfc_age = pfc_all["age"].astype(float)

    with open(CACHE / "pfc_peaks_to_hip_tiles_M.pkl", "rb") as f:
        M = pickle.load(f)
    keep_idx     = np.where(np.asarray(M.sum(axis=0)).ravel() > 0)[0]
    M_keep       = M[:, keep_idx]
    pfc_tile_sum = np.asarray(pfc_sum @ M_keep, dtype=np.float64)

    hip = build_hip_pseudobulks()
    hip_age = hip["age"]

    with open(CACHE / "pfc_per_ct_tile_pseudobulks.pkl", "rb") as f:
        pfc_ct = pickle.load(f)

    print("Running Ridge…")
    panels = []  # (title, preds, y_test, r, mae)

    # All cells
    X_pfc, _ = normalize_zscore(pfc_tile_sum, np.ones(len(pfc_age), dtype=np.int64))
    X_hip, v  = normalize_zscore(hip["all_cells"]["tile_sum"][:, keep_idx],
                                  hip["all_cells"]["cell_counts"])
    preds, y_test, r, mae, pval, ridge_m = run_ridge(X_pfc, pfc_age, X_hip, hip_age[v])
    panels.append(("All cells", preds, y_test, r, mae))
    print(f"  all_cells: r={r:.3f}")
    log_test("pfc_to_hippocampus", "scatter_tile_zscore_horiz_NR2F2inh",
             analysis="PFC->hippocampus tile Ridge age clock (NR2F2->Inhibitory)",
             test="pearson_correlation_actual_vs_predicted_age", group_a="All cells",
             n_a=int(len(y_test)), statistic=r, effect_size=mae, p_value=pval)
    for rank, i in enumerate(np.argsort(np.abs(ridge_m.coef_))[::-1][:200]):
        log_coef("pfc_to_hippocampus", clock_name="pfc_to_hip_tile_zscore_NR2F2inh_clock",
                 feature=f"tile_{keep_idx[i]}", coefficient=float(ridge_m.coef_[i]),
                 modality="ATAC", cell_type="All cells", rank=rank, model_type="RidgeCV")

    for ct in BROAD_CTS:
        X_pfc_ct, pv = normalize_zscore(pfc_ct["per_ct"][ct]["tile_sum"],
                                        pfc_ct["per_ct"][ct]["cell_counts"])
        y_pfc_ct = pfc_ct["age"][pv]
        X_hip_ct, hv = normalize_zscore(hip["per_ct"][ct]["tile_sum"][:, keep_idx],
                                        hip["per_ct"][ct]["cell_counts"])
        y_hip_ct = hip_age[hv]
        if len(y_pfc_ct) < 5 or len(y_hip_ct) < 3:
            continue
        preds, y_test, r, mae, pval, ridge_m = run_ridge(X_pfc_ct, y_pfc_ct, X_hip_ct, y_hip_ct)
        panels.append((ct, preds, y_test, r, mae))
        print(f"  {ct}: r={r:.3f}")
        log_test("pfc_to_hippocampus", "scatter_tile_zscore_horiz_NR2F2inh",
                 analysis="PFC->hippocampus tile Ridge age clock (NR2F2->Inhibitory)",
                 test="pearson_correlation_actual_vs_predicted_age", group_a=ct,
                 n_a=int(len(y_test)), statistic=r, effect_size=mae, p_value=pval)
        for rank, i in enumerate(np.argsort(np.abs(ridge_m.coef_))[::-1][:200]):
            log_coef("pfc_to_hippocampus", clock_name="pfc_to_hip_tile_zscore_NR2F2inh_clock",
                     feature=f"tile_{keep_idx[i]}", coefficient=float(ridge_m.coef_[i]),
                     modality="ATAC", cell_type=ct, rank=rank, model_type="RidgeCV")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 3.2))

    for ax, (title, preds, y_test, r, mae) in zip(axes, panels):
        scatter_panel(ax, y_test, preds, r, mae, title)

    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle("PFC → Hippocampus ATAC Age Clock  |  5 kb tiles  |  CPM + log1p + z-score  |  NR2F2→Inhibitory",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG / f"scatter_tile_zscore_horiz_NR2F2inh.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
