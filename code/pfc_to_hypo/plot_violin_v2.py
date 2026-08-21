#!/usr/bin/env python3
"""
Train the PFC -> mouse hypothalamus ATAC Ridge clock (intersect_all feature
map: mouse peaks lifted mm10->hg38 and overlapped with PFC peaks) for
all_cells and each matched cell type, and plot predicted age as a violin per
chronological age group (Young/Middle/Aged), with Spearman rho annotated.

Saves figures/violin_intersect_all_v2.{pdf,png}
"""
import pickle
import sys
from pathlib import Path

import numpy as np

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

# numpy<2 compat shim: cache/pfc_peak_pseudobulk_by_ct_native.pkl (shared
# atac_processing_techniques cache) was pickled under numpy>=2.0. numpy
# 1.26.4 (pinned by this repo's reproducibility env) ships a small
# numpy/_core/ forward-compat stub package for exactly this situation, but
# that stub covers multiarray/_multiarray_umath/umath/_dtype/_internal and
# NOT numpy._core.numeric -- and scipy/sklearn eagerly import numpy._core
# at their own import time, so by the time this script's pickle.load() runs,
# `numpy._core` already exists (as the incomplete stub) and a naive
# `if not hasattr(np, "_core")` guard would never fire. Patch in the missing
# `numeric` submodule explicitly. No-op under real numpy>=2.0.
try:
    import numpy._core.numeric  # noqa: F401
except ImportError:
    import numpy._core as _np_core_pkg
    import numpy.core.numeric as _np_core_numeric
    sys.modules["numpy._core.numeric"] = _np_core_numeric
    _np_core_pkg.numeric = _np_core_numeric

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT  = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
FIG   = ROOT / "figures"

PFC_ALL_CACHE = Path(
    "~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl"
)
PFC_CT_CACHE = Path(
    "~/atac_processing_techniques/cache"
    "/pfc_peak_pseudobulk_by_ct_native.pkl"
)

RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))

AGE_MAP   = {"Y": 3, "M": 12, "A": 24}
AGE_ORDER = ["Y", "M", "A"]

# level key → face color
CT_COLORS = {
    "All cells":                 "grey",
    "GLU (→Excitatory)":         "tab:orange",
    "GABA (→Inhibitory)":        "tab:green",
    "Astrocytes (→Astro)":       "tab:red",
    "Immune (→Microglia)":       "tab:purple",
    "NG/OPC (→OPC)":             "tab:brown",
    "Oligodendrocytes (→Oligo)": "tab:blue",
}

MOUSE_TO_PFC_CT = {
    "Astrocytes":       "Astro",
    "Oligodendrocytes": "Oligo",
    "NG/OPC":           "OPC",
    "Immune":           "Microglia",
    "GLU":              "Excitatory",
    "GABA":             "Inhibitory",
}

LEVEL_ORDER = [
    "All cells",
    "Astrocytes (→Astro)",
    "Oligodendrocytes (→Oligo)",
    "NG/OPC (→OPC)",
    "Immune (→Microglia)",
    "GLU (→Excitatory)",
    "GABA (→Inhibitory)",
]


def zscore_norm(raw_sum: np.ndarray, cc: np.ndarray):
    valid = cc > 0
    X = raw_sum[valid].astype(np.float64)
    d = X.sum(1, keepdims=True)
    d = np.where(d < 1, 1.0, d)
    X = np.log1p(X / d * 1e6)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-8
    return (X - mu) / sd, valid


def project(raw: np.ndarray, M: sp.spmatrix) -> np.ndarray:
    return M.T.dot(raw.astype(np.float32).T).T.astype(np.float64)


def fit_predict(X_tr, y_tr, X_te):
    return RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, y_tr).predict(X_te)


def plot_violin(levels_data: dict):
    levels = [lv for lv in LEVEL_ORDER if lv in levels_data]
    n = len(levels)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows), squeeze=False
    )
    rng = np.random.default_rng(42)

    x_ticks = np.arange(3, 25, 3)   # 3, 6, 9, 12, 15, 18, 21, 24

    for idx, level in enumerate(levels):
        ax    = axes[idx // ncols][idx % ncols]
        data  = levels_data[level]
        preds = np.asarray(data["preds"])
        ages  = list(data["ages"])

        present = [a for a in AGE_ORDER if a in ages]
        x_pos   = [AGE_MAP[a] for a in present]
        groups  = [preds[np.array(ages) == a] for a in present]

        color = CT_COLORS.get(level, "steelblue")

        if not groups or max(len(g) for g in groups) == 0:
            ax.set_title(f"{level}\n(no data)", fontsize=12)
            continue

        vp = ax.violinplot(
            groups, positions=x_pos, widths=2.2,
            showmedians=True, showextrema=True,
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.50)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            if part in vp:
                vp[part].set_color("0.35")
                vp[part].set_linewidth(1.4)

        for xp, g in zip(x_pos, groups):
            jitter = rng.uniform(-0.25, 0.25, len(g))
            ax.scatter(
                xp + jitter, g,
                s=36, c=color,
                edgecolors="k", linewidths=0.4,
                zorder=5, alpha=0.95,
            )

        x_num_all = [AGE_MAP[a] for a in ages]
        if len(set(x_num_all)) > 1:
            rho, pv = spearmanr(x_num_all, preds)
            sign = "↑" if rho > 0 else "↓"
            log_test(
                "pfc_to_hypo", "violin_intersect_all_v2",
                analysis="PFC->mouse hypothalamus Ridge clock, predicted age vs chronological group",
                test="spearman_correlation", group_a=level, n_a=len(preds),
                statistic=rho, p_value=pv,
            )
            ax.text(
                0.03, 0.97,
                f"ρ={rho:+.2f}  p={pv:.3f}  {sign}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
            )

        ax.set_xlim(0, 27)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks], fontsize=11)
        ax.set_xlabel("Age (months)", fontsize=12)
        ax.set_ylabel("Predicted age (human yr)", fontsize=12)
        ax.set_title(level, fontsize=12, loc="left")
        ax.tick_params(axis="both", labelsize=11)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "PFC (357 human donors) → Mouse Hypothalamus ATAC\nStrategy: intersect_all",
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG / f"violin_intersect_all_v2.{ext}",
            bbox_inches="tight",
            dpi=150 if ext == "png" else None,
        )
    plt.close(fig)
    print("Saved violin_intersect_all_v2.{pdf,png}")


def main():
    print("Loading PFC caches...")
    with open(PFC_ALL_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    pfc_sum = pfc_all["peak_sum"].astype(np.float64)
    pfc_age = pfc_all["age"].astype(float)
    pfc_cc  = pfc_all["cell_counts"]

    with open(PFC_CT_CACHE, "rb") as f:
        pfc_ct_raw = pickle.load(f)

    print("Loading intersect_all feature map...")
    with open(CACHE / "intersect_all_map.pkl", "rb") as f:
        M_m, pmask = pickle.load(f)

    print("Loading hypo pseudobulks...")
    with open(CACHE / "hypo_pseudobulks.pkl", "rb") as f:
        hypo = pickle.load(f)
    with open(CACHE / "hypo_glu_gaba_pseudobulks.pkl", "rb") as f:
        glu_gaba = pickle.load(f)
    hypo["per_ct"]["GLU"]  = glu_gaba["GLU"]
    hypo["per_ct"]["GABA"] = glu_gaba["GABA"]

    hypo_sum    = hypo["all_cells"]["sum"]
    hypo_cc     = hypo["all_cells"]["cc"]
    age_letters = hypo["age_letters"]

    # PFC all-cells features
    pfc_feat = pfc_sum[:, pmask]
    X_pfc, pfc_valid = zscore_norm(pfc_feat, pfc_cc)
    y_pfc = pfc_age[pfc_valid]

    # Hypo all-cells features
    hypo_feat       = project(hypo_sum, M_m)
    X_hypo, h_valid = zscore_norm(hypo_feat, hypo_cc)
    ages_valid      = [age_letters[i] for i in np.where(h_valid)[0]]

    levels_data = {}

    print("Fitting all cells...")
    preds = fit_predict(X_pfc, y_pfc, X_hypo)
    levels_data["All cells"] = {"preds": preds, "ages": ages_valid}

    for mouse_ct, pfc_ct_name in MOUSE_TO_PFC_CT.items():
        if mouse_ct not in hypo["per_ct"] or pfc_ct_name not in pfc_ct_raw:
            continue

        raw_pfc_ct = pfc_ct_raw[pfc_ct_name].astype(np.float64)
        cc_pfc_ct  = (raw_pfc_ct.sum(1) > 0).astype(np.int64)
        pfc_ct_feat = raw_pfc_ct[:, pmask]
        X_pfc_ct, valid_pfc_ct = zscore_norm(pfc_ct_feat, cc_pfc_ct)
        y_pfc_ct = pfc_age[valid_pfc_ct]

        raw_hypo_ct  = hypo["per_ct"][mouse_ct]["sum"]
        cc_hypo_ct   = hypo["per_ct"][mouse_ct]["cc"]
        hypo_ct_feat = project(raw_hypo_ct, M_m)
        X_hypo_ct, valid_hypo_ct = zscore_norm(hypo_ct_feat, cc_hypo_ct)
        ages_ct = [age_letters[i] for i in np.where(valid_hypo_ct)[0]]

        if X_pfc_ct.shape[0] < 5 or X_hypo_ct.shape[0] < 3:
            continue

        print(f"Fitting {mouse_ct}→{pfc_ct_name}...")
        preds_ct  = fit_predict(X_pfc_ct, y_pfc_ct, X_hypo_ct)
        level_key = f"{mouse_ct} (→{pfc_ct_name})"
        levels_data[level_key] = {"preds": preds_ct, "ages": ages_ct}

    print("Plotting...")
    plot_violin(levels_data)


if __name__ == "__main__":
    main()
