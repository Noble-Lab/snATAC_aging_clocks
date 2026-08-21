#!~/micromamba/envs/sat/bin/python
"""Violin plot of NFKBIA peak (chr14:35341586-35342086) accessibility across groups."""

import pickle
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

ROOT  = Path(__file__).resolve().parent
CACHE = ROOT / "cache_v4"
FIG   = ROOT / "figures_v4"

GROUP_ORDER = ["Young WT", "Young SIRT6", "Old WT", "Old SIRT6"]
GROUP_COLORS = {
    "Young WT":    "#4393C3",
    "Young SIRT6": "#92C5DE",
    "Old WT":      "#D6604D",
    "Old SIRT6":   "#762A83",
}

def sample_to_group(col):
    if col.startswith("young_wt"):    return "Young WT"
    if col.startswith("young_sirt6"): return "Young SIRT6"
    if col.startswith("old_wt"):      return "Old WT"
    if col.startswith("old_sirt6"):   return "Old SIRT6"
    return "Unknown"

# Load cached peak matrices (unique_pfc maps local col → global PFC peak idx)
with open(CACHE / "gse294103_peak_matrices.pkl", "rb") as f:
    unique_pfc, X_sirt6, sample_cols = pickle.load(f)

# NFKBIA peak: global PFC idx 98775 → chr14:35341586-35342086
target_global = 98775
local_idx = np.where(unique_pfc == target_global)[0][0]

vals = X_sirt6[:, local_idx]

df = pd.DataFrame({
    "sample": sample_cols,
    "group":  [sample_to_group(s) for s in sample_cols],
    "accessibility": vals,
})

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
rng = np.random.default_rng(0)

for xi, grp in enumerate(GROUP_ORDER):
    sub  = df[df["group"] == grp]["accessibility"].values
    col  = GROUP_COLORS[grp]

    # violin
    vp = ax.violinplot(sub, positions=[xi], widths=0.6,
                       showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor("none")

    # median line
    ax.hlines(np.median(sub), xi - 0.22, xi + 0.22, colors=col, linewidths=2.0, zorder=4)

    # jittered points
    jitter = rng.uniform(-0.12, 0.12, size=len(sub))
    ax.scatter(xi + jitter, sub, color=col, edgecolors="white",
               linewidths=0.5, s=45, zorder=5)

# Significance: Old WT vs Old SIRT6
ow = df[df["group"] == "Old WT"]["accessibility"].values
os = df[df["group"] == "Old SIRT6"]["accessibility"].values
stat, pval = mannwhitneyu(ow, os, alternative="two-sided")
print(f"Old WT vs Old SIRT6: U={stat:.0f}, p={pval:.4f}, n={len(ow)},{len(os)}")
log_test("pfc_to_perturbation", "nfkbia_peak_violin", analysis="NFKBIA peak accessibility, Old WT vs Old SIRT6",
         test="mannwhitneyu", group_a="Old WT", group_b="Old SIRT6",
         n_a=len(ow), n_b=len(os), statistic=stat, p_value=pval)

yw = df[df["group"] == "Young WT"]["accessibility"].values
ys = df[df["group"] == "Young SIRT6"]["accessibility"].values
stat2, pval2 = mannwhitneyu(yw, ys, alternative="two-sided")
print(f"Young WT vs Young SIRT6: U={stat2:.0f}, p={pval2:.4f}, n={len(yw)},{len(ys)}")
log_test("pfc_to_perturbation", "nfkbia_peak_violin", analysis="NFKBIA peak accessibility, Young WT vs Young SIRT6",
         test="mannwhitneyu", group_a="Young WT", group_b="Young SIRT6",
         n_a=len(yw), n_b=len(ys), statistic=stat2, p_value=pval2)

ax.set_xticks(range(len(GROUP_ORDER)))
ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("ATAC-seq accessibility (norm. reads)", fontsize=11)
ax.set_title("NFKBIA locus\nchr14:35,341,586–35,342,086", fontsize=12)
ax.set_xlim(-0.6, len(GROUP_ORDER) - 0.4)

plt.tight_layout()
out = FIG / "nfkbia_peak_violin.pdf"
fig.savefig(out, bbox_inches="tight")
out_png = FIG / "nfkbia_peak_violin.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"Saved: {out_png}")
