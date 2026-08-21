"""
All-peaks per-donor z-score age clock: PFC internal validation.

5-fold cross-validation on all 357 PFC donors, all cells and each of 6 broad
cell types.

Feature space: native PFC peaks (521 217) for both all-cells and per-CT.
  Cache: atac_processing_techniques/cache/pfc_peak_pseudobulk_by_ct_native.pkl
  Built on-the-fly from final_atac_data.h5ad if not present.

Normalisation:
  CPM+log1p → per-donor z-score (each row over its own features).

Outputs:
  results/allpeaks_cv_results.csv
  figures/allpeaks_5foldCV.pdf/.png
"""
import gc, time, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _numpy2_compat import load_pickle_compat

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

ATAC_ROOT = Path("~/atac_processing_techniques")
CACHE     = ATAC_ROOT / "cache"
ROOT      = Path(__file__).resolve().parent.parent
RESULTS   = ROOT / "results"
FIG       = ROOT / "figures"
for d in (RESULTS, FIG):
    d.mkdir(parents=True, exist_ok=True)

PFC_ATAC_H5AD = Path("~/data_back/PFC_brain_multiome/final_atac_data.h5ad")
PFC_RNA_H5AD  = Path("~/data_back/PFC_brain_multiome/final_rna_data.h5ad")
NATIVE_CT_PKL = CACHE / "pfc_peak_pseudobulk_by_ct_native.pkl"

ALPHAS = tuple(np.logspace(-2, 6, 30))
N_FOLDS = 5
CHUNK_SIZE = 5000
CELL_TYPES = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]

PFC_CT_MAP = {
    "ExN": "Excitatory", "InN": "Inhibitory",
    "Oligo": "Oligo", "Astro": "Astro",
    "MG": "Microglia", "OPC": "OPC",
}

CT_COLORS = {
    "All cells":  "#444444",
    "Excitatory": "#E64B35",
    "Inhibitory": "#4DBBD5",
    "Oligo":      "#00A087",
    "Astro":      "#3C5488",
    "Microglia":  "#F39B7F",
    "OPC":        "#8491B4",
}


# ── helpers ────────────────────────────────────────────────────────────────
def cpm_log1p(X: np.ndarray) -> np.ndarray:
    d = X.sum(1, keepdims=True)
    d = np.where(d < 1, 1.0, d)
    return np.log1p(X / d * 1e6)


def sample_zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


def norm(X: np.ndarray) -> np.ndarray:
    return sample_zscore(cpm_log1p(X.astype(np.float32)))


def pearson_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def fit_predict(X_tr, y_tr, X_te):
    """RidgeCV (GCV on training set) → predictions on X_te."""
    m = RidgeCV(alphas=ALPHAS)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def five_fold_cv(X, y, label=""):
    """Return (actual, predicted) arrays from 5-fold OOF predictions."""
    preds = np.full(len(y), np.nan)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(kf.split(X)):
        print(f"    {label} fold {fold+1}/{N_FOLDS}…", flush=True)
        preds[va] = fit_predict(X[tr], y[tr], X[va])
    return y, preds


# ── build native per-CT cache (single h5ad pass) ──────────────────────────
def build_pfc_by_ct_native(donors_ref, n_pfc_peaks):
    """
    Read PFC ATAC h5ad in chunks → accumulate per-CT donor×peak sums
    in native PFC peak space (521 217 features).
    Returns dict: CT → float32 array (N_donors, n_pfc_peaks).
    """
    print(f"  [build] Loading RNA for age map…")
    a_rna = ad.read_h5ad(PFC_RNA_H5AD, backed="r")
    sid_age = (a_rna.obs[["SampleID", "Age"]]
               .groupby("SampleID")["Age"].first())
    a_rna.file.close()

    print(f"  [build] Opening ATAC h5ad ({PFC_ATAC_H5AD})…")
    a = ad.read_h5ad(PFC_ATAC_H5AD, backed="r")
    obs = a.obs.copy()
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["Age"]       = obs["sample_id"].map(sid_age).astype(float)
    obs["broad_ct"]  = obs["cell_type"].map(PFC_CT_MAP)
    obs = obs[obs["Age"].notna() & obs["broad_ct"].notna()]
    print(f"  [build] {len(obs):,} cells after filter")

    N    = len(donors_ref)
    d2i  = {d: i for i, d in enumerate(donors_ref)}
    acc  = {ct: np.zeros((N, n_pfc_peaks), dtype=np.float32)
            for ct in CELL_TYPES}

    cell_ids = obs.index.values
    n_cells  = len(cell_ids)
    t0 = time.time()
    for start in range(0, n_cells, CHUNK_SIZE):
        end     = min(start + CHUNK_SIZE, n_cells)
        ids     = cell_ids[start:end]
        sub_obs = obs.loc[ids]
        valid   = sub_obs["sample_id"].isin(d2i)
        if not valid.any():
            continue
        ids_v   = ids[valid.values]
        dc_v    = sub_obs.loc[ids_v, "sample_id"].map(d2i).values
        bcts_v  = sub_obs.loc[ids_v, "broad_ct"].values
        sub     = a[ids_v].to_memory()
        X       = sub.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.astype(np.float32)
        for ct in CELL_TYPES:
            ct_mask = bcts_v == ct
            if not ct_mask.any():
                continue
            dc_ct = dc_v[ct_mask]
            X_ct  = X[ct_mask]
            csz   = int(ct_mask.sum())
            D     = sp.csr_matrix(
                (np.ones(csz, np.float32),
                 (dc_ct.astype(np.int64), np.arange(csz, dtype=np.int64))),
                shape=(N, csz))
            DX = D @ X_ct
            acc[ct] += DX.toarray() if sp.issparse(DX) else np.asarray(DX)
        del sub, X
        gc.collect()
        if (end // CHUNK_SIZE) % 100 == 0:
            print(f"  [build] {end}/{n_cells} cells  "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
    a.file.close()
    print(f"  [build] Done in {(time.time()-t0)/60:.1f}m")
    return acc


# ── load caches ────────────────────────────────────────────────────────────
print("Loading all-cells cache…")
with open(CACHE / "pfc_peak_pseudobulk.pkl", "rb") as f:
    pfc_all = pickle.load(f)

donors  = np.array(pfc_all["donors"])
age     = pfc_all["age"].astype(np.float64)
n_peaks = pfc_all["peak_sum"].shape[1]     # 521 217

# ── build / load native per-CT cache ──────────────────────────────────────
if NATIVE_CT_PKL.exists():
    print(f"Loading native per-CT cache from {NATIVE_CT_PKL}…")
    pfc_ct_raw = load_pickle_compat(NATIVE_CT_PKL)
else:
    print("Native per-CT cache not found — building from h5ad…")
    pfc_ct_raw = build_pfc_by_ct_native(donors.tolist(), n_peaks)
    with open(NATIVE_CT_PKL, "wb") as f:
        pickle.dump(pfc_ct_raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {NATIVE_CT_PKL}")

# ── normalise (once, before any splitting) ─────────────────────────────────
print("Normalising all-cells matrix…")
X_all = norm(pfc_all["peak_sum"])
print(f"  X_all: {X_all.shape}")

print("Normalising per-CT matrices…")
X_ct = {}
for ct in CELL_TYPES:
    X_ct[ct] = norm(pfc_ct_raw[ct])
    print(f"  {ct}: {X_ct[ct].shape}")


# ══════════════════════════════════════════════════════════════════════════
# 1. 5-fold CV
# ══════════════════════════════════════════════════════════════════════════
print("\n=== 5-fold CV ===")
cv_actual, cv_pred, cv_labels = [], [], []

print("  All cells…")
act, pred = five_fold_cv(X_all, age, "all-cells")
cv_actual.append(act); cv_pred.append(pred); cv_labels.append("All cells")
print(f"  All cells  r={pearson_r(act, pred):.3f}")

for ct in CELL_TYPES:
    print(f"  {ct}…")
    act, pred = five_fold_cv(X_ct[ct], age, ct)
    cv_actual.append(act); cv_pred.append(pred); cv_labels.append(ct)
    print(f"  {ct}  r={pearson_r(act, pred):.3f}")


# ══════════════════════════════════════════════════════════════════════════
# Save results CSV
# ══════════════════════════════════════════════════════════════════════════
rows = []
for lbl, act, pred in zip(cv_labels, cv_actual, cv_pred):
    rows.append(dict(eval="5fold_CV", cell_type=lbl,
                     r=pearson_r(act, pred),
                     mae=float(np.nanmean(np.abs(act - pred))),
                     n=int(np.isfinite(pred).sum())))

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "allpeaks_cv_results.csv", index=False)
print(f"\nSaved {RESULTS / 'allpeaks_cv_results.csv'}")
print(df.to_string(index=False))

# Log Pearson r + p-value for each cell type underlying figures/allpeaks_5foldCV.pdf.
for lbl, act, pred in zip(cv_labels, cv_actual, cv_pred):
    m = np.isfinite(act) & np.isfinite(pred)
    r, p = (pearsonr(act[m], pred[m]) if m.sum() >= 3 else (np.nan, np.nan))
    log_test(
        "pfc_internal_valid", "allpeaks_5foldCV", analysis="Ridge age clock (5fold_CV)",
        test="pearson_correlation_actual_vs_predicted_age",
        group_a="5fold_CV", group_b=lbl, n_a=int(m.sum()), statistic=r, p_value=p,
        notes="all-peaks Ridge clock, per-donor z-scored CPM",
    )


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════
def scatter_panel(ax, actual, pred, label, color):
    act = np.asarray(actual, float)
    prd = np.asarray(pred, float)
    m   = np.isfinite(act) & np.isfinite(prd)
    r   = float(pearsonr(act[m], prd[m])[0]) if m.sum() >= 3 else np.nan
    mae = float(np.nanmean(np.abs(act[m] - prd[m])))
    lo  = min(act[m].min(), prd[m].min()) - 3
    hi  = max(act[m].max(), prd[m].max()) + 3
    ax.scatter(act[m], prd[m], s=14, alpha=0.6, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual age", fontsize=8)
    ax.set_ylabel("Predicted age", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)
    ax.tick_params(labelsize=7)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nMAE = {mae:.1f} yr\nn = {m.sum()}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))


def make_figure(actual_list, pred_list, label_list, suptitle, fname):
    n = len(label_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = axes.flatten()
    for i, (act, pred, lbl) in enumerate(zip(actual_list, pred_list, label_list)):
        scatter_panel(axes_flat[i], act, pred, lbl, CT_COLORS.get(lbl, "#444444"))
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG / f"{fname}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved {p}")
    plt.close(fig)


print("\nPlotting…")
make_figure(cv_actual, cv_pred, cv_labels,
            "PFC 5-fold CV — all native peaks, per-donor z-score",
            "allpeaks_5foldCV")

print("\nDone.")
