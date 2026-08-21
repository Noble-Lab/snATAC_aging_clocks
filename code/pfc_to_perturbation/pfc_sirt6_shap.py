#!~/micromamba/envs/sat/bin/python
"""
Linear SHAP analysis for PFC→GSE294103 SIRT6 peak-level clock
==============================================================
Produces 3 figures:
  Fig 1: Top 20 peaks by mean|SHAP| in WT donors (left) and SIRT6 donors (right)
  Fig 2: Top 20 peaks ranked by (SIRT6 - WT) mean|SHAP| difference
  Fig 3: Top 20 Reactome pathways enriched in genes near top-2000 SIRT6-enriched peaks

External large-file dependency (kept at a fixed absolute path, not inside
this repo): ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
(238MB). Everything else (cache_v4/, figures_v4/, results_v4/, logs/,
reference/) is local to this script's own directory.
"""

import logging
import pickle
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test, log_coef

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeCV

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
FIG     = ROOT / "figures_v4"
RESULTS = ROOT / "results_v4"
LOG_DIR = ROOT / "logs"
CACHE   = ROOT / "cache_v4"

ATAC_META_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
TSS_FILE      = ROOT / "reference" / "hg38_gene_tss.tsv"  # bundled copy (892KB)
PEAK_MAT_PKL  = CACHE / "gse294103_peak_matrices.pkl"
SHAP_PKL      = CACHE / "gse294103_shap_values.pkl"

RIDGE_ALPHAS = tuple(np.logspace(-2, 8, 40))
N_TOP_SHAP   = 20
N_DIFF_PEAKS = 2000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sirt6_shap.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

# ── CPM + log1p, then per-donor z-score ─────────────────────────────────────
def normalize(X: np.ndarray) -> np.ndarray:
    X  = np.asarray(X, dtype=np.float64)
    d  = X.sum(1, keepdims=True).clip(1e-9)
    X  = np.log1p(X / d * 1e6)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


# ── Sample group membership ────────────────────────────────────────────────────
def sample_group(col: str) -> str:
    if "sirt6" in col: return "SIRT6"
    return "WT"


# ── Load data & train ──────────────────────────────────────────────────────────
def load_and_train():
    log.info("Loading cached peak matrices …")
    with open(PEAK_MAT_PKL, "rb") as f:
        unique_pfc, X_sirt6, sample_cols = pickle.load(f)

    log.info("Loading PFC training data …")
    with open(ATAC_META_PKL, "rb") as f:
        meta = pickle.load(f)
    pfc_ages     = np.array(meta["age"])
    pfc_peak_sum = meta["peak_sum"]
    peak_names   = meta["peak_names"]

    X_pfc      = pfc_peak_sum[:, unique_pfc].astype(np.float64)
    X_pfc_norm = normalize(X_pfc)
    X_sirt6_norm = normalize(X_sirt6.astype(np.float64))

    log.info(f"Training Ridge on {X_pfc_norm.shape[1]:,} features …")
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge.fit(X_pfc_norm, pfc_ages)
    log.info(f"  alpha={ridge.alpha_:.3g}")
    peak_names_used = [peak_names[i] for i in unique_pfc]
    for rank, i in enumerate(np.argsort(np.abs(ridge.coef_))[::-1][:200]):
        log_coef("pfc_to_perturbation", clock_name="gse294103_sirt6_shap_ridge_clock",
                 feature=peak_names_used[i], coefficient=float(ridge.coef_[i]),
                 modality="ATAC", cell_type="All cells", rank=rank, model_type="RidgeCV")

    return ridge, X_pfc_norm, X_sirt6_norm, unique_pfc, sample_cols, peak_names


# ── Linear SHAP ───────────────────────────────────────────────────────────────
def compute_shap(ridge, X_pfc_norm, X_sirt6_norm, sample_cols):
    if SHAP_PKL.exists():
        log.info("Loading cached SHAP values …")
        with open(SHAP_PKL, "rb") as f:
            return pickle.load(f)

    log.info("Computing linear SHAP values …")
    explainer  = shap.LinearExplainer(ridge, X_pfc_norm, feature_perturbation="interventional")
    shap_vals  = explainer.shap_values(X_sirt6_norm)   # (28, 50633)
    log.info(f"  SHAP matrix: {shap_vals.shape}")

    with open(SHAP_PKL, "wb") as f:
        pickle.dump(shap_vals, f, protocol=4)
    return shap_vals


# ── Nearest gene annotation ────────────────────────────────────────────────────
def build_tss_index(tss_df: pd.DataFrame) -> dict:
    """chr → (sorted tss array, gene_name array)."""
    idx = {}
    for ch, grp in tss_df.groupby("chr"):
        grp = grp.sort_values("tss")
        idx[ch] = (grp["tss"].values, grp["gene_name"].values)
    return idx


def nearest_gene(ch: str, mid: int, tss_idx: dict) -> str:
    if ch not in tss_idx:
        return "unknown"
    tss_arr, name_arr = tss_idx[ch]
    i = np.searchsorted(tss_arr, mid)
    candidates = []
    if i < len(tss_arr):
        candidates.append((abs(tss_arr[i] - mid), name_arr[i]))
    if i > 0:
        candidates.append((abs(tss_arr[i - 1] - mid), name_arr[i - 1]))
    return min(candidates, key=lambda x: x[0])[1]


def annotate_peaks(peak_names_subset, tss_idx):
    labels = []
    for name in peak_names_subset:
        ch, coords = name.split(":")
        s, e = coords.split("-")
        mid  = (int(s) + int(e)) // 2
        gene = nearest_gene(ch, mid, tss_idx)
        labels.append(f"{gene}\n({name})")
    return labels


# ── Mean |SHAP| per group per feature ─────────────────────────────────────────
def group_mean_abs_shap(shap_vals, sample_cols):
    abs_shap = np.abs(shap_vals)
    wt_mask   = np.array([sample_group(s) == "WT"    for s in sample_cols])
    sirt6_mask = np.array([sample_group(s) == "SIRT6" for s in sample_cols])
    wt_mean    = abs_shap[wt_mask].mean(0)
    sirt6_mean = abs_shap[sirt6_mask].mean(0)
    return wt_mean, sirt6_mean


# ── Figure 1: Top 20 peaks by mean|SHAP| in WT and SIRT6 ──────────────────────
def plot_top20_shap(wt_mean, sirt6_mean, peak_labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mean_shap, group, color in [
        (axes[0], wt_mean,    "WT",    "#D6604D"),
        (axes[1], sirt6_mean, "SIRT6", "#762A83"),
    ]:
        top_idx = np.argsort(mean_shap)[-N_TOP_SHAP:][::-1]
        vals    = mean_shap[top_idx]
        labels  = [peak_labels[i] for i in top_idx]

        y = np.arange(N_TOP_SHAP)
        ax.barh(y, vals, color=color, alpha=0.75, edgecolor="0.3", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| (age yr)", fontsize=9)
        ax.set_title(f"Top 20 peaks — {group} donors\n(mean absolute SHAP)", fontsize=10)
        for xi, (bar_v, lbl) in enumerate(zip(vals, labels)):
            ax.text(bar_v * 1.01, xi, f"{bar_v:.3f}", va="center", fontsize=6.5, color="0.25")

    fig.suptitle("PFC clock linear SHAP — GSE294103 SIRT6 OE\n"
                 "Peak-level features (mm10→hg38 liftover)", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "sirt6_shap_top20_wt_sirt6")


# ── Figure 2: Top 20 peaks by (SIRT6 - WT) mean|SHAP| difference ──────────────
def plot_diff_top20(wt_mean, sirt6_mean, peak_labels):
    diff    = sirt6_mean - wt_mean
    top_idx = np.argsort(diff)[-N_TOP_SHAP:][::-1]

    vals_diff  = diff[top_idx]
    vals_wt    = wt_mean[top_idx]
    vals_sirt6 = sirt6_mean[top_idx]
    labels     = [peak_labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(N_TOP_SHAP)
    w = 0.35

    ax.barh(y - w/2, vals_wt,    w, label="WT",    color="#D6604D", alpha=0.75, edgecolor="0.3", linewidth=0.5)
    ax.barh(y + w/2, vals_sirt6, w, label="SIRT6", color="#762A83", alpha=0.75, edgecolor="0.3", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (age yr)", fontsize=9)
    ax.set_title(
        "Top 20 peaks by SIRT6 − WT mean |SHAP| difference\n"
        "(peaks more predictive in SIRT6 than WT donors)",
        fontsize=10, pad=6,
    )
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.axvline(0, color="0.5", linewidth=0.7, linestyle="--")

    fig.tight_layout()
    _save(fig, "sirt6_shap_diff_top20")
    return top_idx


# ── Figure 3: Reactome enrichment for top-2000 SIRT6-enriched peaks ───────────
def plot_reactome(diff, peak_names_full, tss_idx):
    top2000_idx = np.argsort(diff)[-N_DIFF_PEAKS:][::-1]

    genes = set()
    for i in top2000_idx:
        name = peak_names_full[i]
        ch, coords = name.split(":")
        s, e = coords.split("-")
        mid  = (int(s) + int(e)) // 2
        g    = nearest_gene(ch, mid, tss_idx)
        if g != "unknown":
            genes.add(g)
    genes = list(genes)
    log.info(f"Unique genes near top-{N_DIFF_PEAKS} SIRT6-enriched peaks: {len(genes)}")

    pd.Series(sorted(genes)).to_csv(RESULTS / "sirt6_enriched_genes.txt", index=False, header=False)

    log.info("Running Reactome enrichment via Enrichr …")
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets="Reactome_2022",
            outdir=None,
            verbose=False,
        )
        df_enr = enr.results.copy()
    except Exception as e:
        log.warning(f"Enrichr failed: {e}. Trying offline Reactome_2016 …")
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets="Reactome_2016",
            outdir=None,
            verbose=False,
        )
        df_enr = enr.results.copy()

    df_enr = df_enr.sort_values("Adjusted P-value")
    df_enr.to_csv(RESULTS / "sirt6_reactome_enrichment.csv", index=False)
    log.info(f"  Top pathway: {df_enr.iloc[0]['Term']} (adj-p={df_enr.iloc[0]['Adjusted P-value']:.3e})")
    for _, r in df_enr.iterrows():
        log_test("pfc_to_perturbation", "sirt6_shap_diff_top20_pct_dumbbell_labeled_mm10",
                 analysis="Enrichr (Reactome) on top SIRT6-diff SHAP genes",
                 test="enrichr_fisher_exact", group_a="Reactome_2016", group_b=r["Term"],
                 statistic=r.get("Odds Ratio"), p_value=r.get("P-value"), q_value=r.get("Adjusted P-value"))

    top20 = df_enr.head(N_TOP_SHAP).copy()
    top20["-log10(adj-p)"] = -np.log10(top20["Adjusted P-value"].clip(1e-300))

    # shorten long pathway names
    top20["Term_short"] = top20["Term"].str.replace(r" R-HSA-\d+$", "", regex=True).str[:70]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top20))
    bars = ax.barh(y, top20["-log10(adj-p)"].values,
                   color="#4D9221", alpha=0.75, edgecolor="0.3", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top20["Term_short"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("-log₁₀(adj. p-value)", fontsize=9)
    ax.axvline(-np.log10(0.05), color="tomato", linewidth=1.2,
               linestyle="--", label="adj-p = 0.05")

    for xi, (bar_v, row) in enumerate(zip(top20["-log10(adj-p)"].values, top20.itertuples())):
        n_overlap = str(row._asdict().get("Overlap", "")).split("/")[0]
        ax.text(bar_v + 0.05, xi, f"n={n_overlap}", va="center", fontsize=6.5, color="0.3")

    ax.set_title(
        f"Reactome enrichment: genes near top-{N_DIFF_PEAKS} SIRT6-enriched peaks\n"
        f"({len(genes)} unique genes, ranked by SIRT6−WT mean |SHAP| difference)",
        fontsize=9.5, pad=6,
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, "sirt6_reactome_top20")

    return df_enr


# ── Save helper ────────────────────────────────────────────────────────────────
def _save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        log.info(f"Saved {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ridge, X_pfc_norm, X_sirt6_norm, unique_pfc, sample_cols, all_peak_names = load_and_train()

    shap_vals = compute_shap(ridge, X_pfc_norm, X_sirt6_norm, sample_cols)

    # peak names for the 50,633 features used
    peak_names_used = [all_peak_names[i] for i in unique_pfc]

    log.info("Loading TSS annotation …")
    tss_df  = pd.read_csv(TSS_FILE, sep="\t")
    tss_idx = build_tss_index(tss_df)

    log.info(f"Annotating {len(peak_names_used):,} peaks with nearest gene …")
    peak_labels = annotate_peaks(peak_names_used, tss_idx)

    wt_mean, sirt6_mean = group_mean_abs_shap(shap_vals, sample_cols)

    log.info(f"WT   top feature mean|SHAP|: {wt_mean.max():.4f}")
    log.info(f"SIRT6 top feature mean|SHAP|: {sirt6_mean.max():.4f}")

    plot_top20_shap(wt_mean, sirt6_mean, peak_labels)
    diff_top_idx = plot_diff_top20(wt_mean, sirt6_mean, peak_labels)

    diff = sirt6_mean - wt_mean
    plot_reactome(diff, peak_names_used, tss_idx)

    log.info("Done.")


if __name__ == "__main__":
    main()
