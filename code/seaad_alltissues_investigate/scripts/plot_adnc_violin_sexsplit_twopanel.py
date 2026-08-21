"""Two-panel split-sex violin: (Not AD + Low + Intermediate) vs High ADNC.

Each panel shows cell types on x-axis, residuals on y-axis, with female/male
half-violins. Mann-Whitney U test asterisks mark significant sex differences.

Produces:
  dlpfc_xds_pfc357_peaks_all_adnc_celltype_violin_sexsplit_twopanel.pdf
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT       = Path(__file__).resolve().parent.parent
RES        = ROOT / "results"
FIG        = ROOT / "figures"
SEAAD_META = Path("~/age_accel_per_cell_type/cache/seaad_pseudobulks_by_ct.pkl")

ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
ADNC_NUM   = {v: i for i, v in enumerate(ADNC_ORDER)}

SEX_PALETTE = {"Female": "#c0392b", "Male": "#2980b9"}

LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]
LABEL_DISPLAY = {
    "all_cells": "All cells",
    "Exc":       "Excitatory",
    "Inh":       "Inhibitory",
    "Oligo":     "Oligodendrocyte",
    "Astro":     "Astrocyte",
    "Mic":       "Microglia",
    "OPC":       "OPC",
}

# Panel definitions: (title, list of ADNC stages to pool)
PANELS = [
    ("Not AD / Low / Intermediate", ["Not AD", "Low", "Intermediate"]),
    ("High",                         ["High"]),
]


def load_with_meta(label: str, sex_series: pd.Series,
                   adnc_series: pd.Series) -> pd.DataFrame | None:
    path = RES / f"xds_pfc357_peaks_{label}_all.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["sex"]      = df["donor_id"].map(sex_series)
    df["adnc"]     = df["donor_id"].map(adnc_series)
    df["adnc_num"] = df["adnc"].map(ADNC_NUM)
    return df


def half_violin(ax, data, pos, side, color, width=0.38):
    if len(data) < 3:
        return
    ys = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
    try:
        kde = stats.gaussian_kde(data, bw_method="scott")
        density = kde(ys)
    except Exception:
        return
    density = density / density.max() * width

    if side == "left":
        ax.fill_betweenx(ys, pos - density, pos, alpha=0.65, color=color, linewidth=0)
        ax.plot(pos - density, ys, color=color, lw=0.8)
    else:
        ax.fill_betweenx(ys, pos, pos + density, alpha=0.65, color=color, linewidth=0)
        ax.plot(pos + density, ys, color=color, lw=0.8)

    med = np.median(data)
    ax.hlines(med, pos - (width if side == "left" else 0),
              pos + (width if side == "right" else 0),
              colors="white", lw=1.8, zorder=5)
    ax.hlines(med, pos - (width if side == "left" else 0),
              pos + (width if side == "right" else 0),
              colors=color, lw=0.9, linestyles="--", zorder=6)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)
    if pooled_var == 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def pval_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_panel(ax, title, adnc_stages, data, show_legend=False, fontsize=11):
    rng = np.random.default_rng(42)
    present_labels = [lb for lb in LABELS if lb in data]

    # First pass: draw violins and collect per-cell-type stats
    ct_stats = []  # (xi, grp_f, grp_m)
    for xi, label in enumerate(present_labels):
        df  = data[label]
        sub = df[df["adnc"].isin(adnc_stages)]

        grp_f = sub.loc[sub["sex"] == "Female", "residual"].dropna().values
        grp_m = sub.loc[sub["sex"] == "Male",   "residual"].dropna().values
        ct_stats.append((xi, grp_f, grp_m))

        for sex, side, grp in [("Female", "left", grp_f), ("Male", "right", grp_m)]:
            if len(grp) == 0:
                continue
            color = SEX_PALETTE[sex]
            half_violin(ax, grp, xi, side, color, width=0.38)
            jit    = rng.uniform(-0.06, 0.06, len(grp))
            offset = -0.10 if side == "left" else 0.10
            ax.scatter(xi + offset + jit, grp, color=color,
                       s=16, alpha=0.85, zorder=4,
                       edgecolors="white", linewidths=0.3)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(present_labels)))
    ax.set_xticklabels(
        [LABEL_DISPLAY[lb] for lb in present_labels],
        rotation=20, ha="right", fontsize=fontsize - 1,
    )
    ax.set_ylabel("OLS-corrected brain-age residual (years)", fontsize=fontsize - 1)
    ax.set_xlim(-0.6, len(present_labels) - 0.4)
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")

    if show_legend:
        handles = [mpatches.Patch(color=SEX_PALETTE[s], label=s) for s in ["Female", "Male"]]
        ax.legend(handles=handles, fontsize=fontsize - 1, loc="upper left", framealpha=0.8)

    # Extend y-axis headroom for effect-size labels, then annotate
    y_bot, y_top_cur = ax.get_ylim()
    data_tops = []
    for xi, grp_f, grp_m in ct_stats:
        tops = [g.max() for g in (grp_f, grp_m) if len(g) > 0]
        data_tops.append(max(tops) if tops else y_top_cur)
    y_top_new = max(y_top_cur, max(data_tops) + 3.5)
    ax.set_ylim(y_bot, y_top_new)

    for xi, grp_f, grp_m in ct_stats:
        if len(grp_f) < 2 or len(grp_m) < 2:
            continue
        d = cohens_d(grp_f, grp_m)
        _, p = stats.mannwhitneyu(grp_f, grp_m, alternative="two-sided")
        stars = pval_to_stars(p)
        log_test(
            "seaad_alltissues_investigate", "dlpfc_xds_pfc357_peaks_all_adnc_celltype_violin_sexsplit_twopanel",
            analysis=f"{title}: brain-age residual, Female vs Male",
            test="mannwhitneyu", group_a="Female", group_b="Male",
            n_a=len(grp_f), n_b=len(grp_m), effect_size=d, p_value=p,
            notes=f"cell_type={present_labels[xi]}",
        )
        tops = [g.max() for g in (grp_f, grp_m) if len(g) > 0]
        y_anchor = max(tops) + 0.4
        d_str  = f"d={d:+.2f}" if not np.isnan(d) else "d=n/a"
        label  = f"{d_str}{stars}" if stars else d_str
        color  = "black" if not stars else "#7d3c98"
        ax.text(xi, y_anchor, label, ha="center", va="bottom",
                fontsize=fontsize - 2, color=color, zorder=7,
                fontweight="bold" if stars else "normal")

    # Sample-size annotations
    y_bot = ax.get_ylim()[0]
    for xi, label in enumerate(present_labels):
        df  = data[label]
        sub = df[df["adnc"].isin(adnc_stages)]
        nf  = (sub["sex"] == "Female").sum()
        nm  = (sub["sex"] == "Male").sum()
        ax.text(xi - 0.18, y_bot + 0.3, f"n={nf}", ha="center", va="bottom",
                fontsize=max(4, fontsize - 7), color=SEX_PALETTE["Female"])
        ax.text(xi + 0.18, y_bot + 0.3, f"n={nm}", ha="center", va="bottom",
                fontsize=max(4, fontsize - 7), color=SEX_PALETTE["Male"])


def main() -> None:
    # The donor-metadata cache was pickled under numpy>=2.0; _seaad_meta_compat.py loads
    # a numpy-version-agnostic copy of it instead of unpickling it directly.
    from _seaad_meta_compat import load_seaad_meta
    meta        = load_seaad_meta()
    sex_series  = meta["Sex"]
    adnc_series = meta["Overall AD neuropathological Change"]

    data: dict[str, pd.DataFrame] = {}
    for label in LABELS:
        df = load_with_meta(label, sex_series, adnc_series)
        if df is None or df.empty:
            print(f"  [{label}] no CSV, skipping")
            continue
        data[label] = df

    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 7), sharey=False, squeeze=True)
    fig.suptitle(
        "PFC-357 peaks → DLPFC  ·  residual vs cell type  ·  split by sex\n"
        "d = Cohen's d (female − male)  ·  * p<0.05  ** p<0.01  *** p<0.001  (Mann-Whitney)",
        fontsize=12, y=1.03,
    )

    for ax_idx, (title, stages) in enumerate(PANELS):
        draw_panel(
            axes[ax_idx], title, stages, data,
            show_legend=(ax_idx == 0),
            fontsize=11,
        )

    fig.tight_layout()
    outpath = FIG / "dlpfc_xds_pfc357_peaks_all_adnc_celltype_violin_sexsplit_twopanel.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath.name}")


if __name__ == "__main__":
    main()
