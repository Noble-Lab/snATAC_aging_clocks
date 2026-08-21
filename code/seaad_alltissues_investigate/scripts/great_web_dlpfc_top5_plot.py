"""Plot the top 5 pathways/terms per cell type for each GREAT-web Reactome
tag. Mirrors ~/pfc_internal_valid/scripts/great_web_top5.py.

Loads: results/dlpfc_pathway_enrichment_GREATweb_Reactome_{tag}.csv
Saves: figures/dlpfc_pathway_reactome_GREATweb_top5_{tag}.pdf/.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from great_web_dlpfc_sig_plot import (
    RESULTS, FIG, ALL_LABELS, LABEL_DISPLAY, CT_COLORS,
    clean_name, describe_tag,
)

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
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

TOP_K_PER_LABEL = 5


def make_figure(tag: str) -> None:
    csv_path = RESULTS / f"dlpfc_pathway_enrichment_GREATweb_Reactome_{tag}.csv"
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["name_clean"] = df["name"].apply(clean_name)
    df["label_text"] = (
        df["name_clean"] + " (" +
        df["Hyper_Observed_Gene_Hits"].astype(int).astype(str) + "/" +
        df["Hyper_Total_Genes"].astype(int).astype(str) + ")"
    )
    df["neglog_q"] = -np.log10(df["q_binom"].clip(lower=1e-300))

    for _, r in df.iterrows():
        log_test(
            "seaad_alltissues_investigate", f"dlpfc_pathway_reactome_GREATweb_top5_{tag}",
            analysis=f"GREAT web (great.stanford.edu v3.0.0/hg19) REACTOME enrichment ({tag})",
            test="binomial_region_enrichment", group_a=r.get("label"), group_b=r.get("name"),
            statistic=r.get("Binom_Fold_Enrichment"), p_value=r.get("Binom_Raw_PValue"),
            q_value=r.get("q_binom"), notes=f"tag={tag}",
        )

    rows = []
    n_sig_shown = 0
    for label in ALL_LABELS:
        sub = df[df["label"] == label].sort_values("q_binom", ascending=True).head(TOP_K_PER_LABEL)
        n_sig_shown += int((sub["q_binom"] < 0.05).sum())
        for _, r in sub.iterrows():
            rows.append({
                "label":      label,
                "label_text": r["label_text"],
                "neglog_q":   r["neglog_q"],
                "color":      CT_COLORS[label],
                "sig":        r["q_binom"] < 0.05,
            })
    if not rows:
        print(f"  {tag}: no data.")
        return
    plot_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"{tag}: {len(plot_df)} bars shown (top {TOP_K_PER_LABEL}/cell type), "
          f"{n_sig_shown} of them q_binom<0.05")

    n_bars = len(plot_df)
    bar_h  = 0.32
    fig_h  = max(3.0, n_bars * bar_h + 1.8)
    max_x  = plot_df["neglog_q"].max()

    fig, ax = plt.subplots(figsize=(9, fig_h))
    y_pos = np.arange(n_bars, dtype=float)

    alphas = np.where(plot_df["sig"].values, 0.85, 0.35)
    for y, val, color, a in zip(y_pos, plot_df["neglog_q"].values, plot_df["color"].values, alphas):
        ax.barh(y, val, color=color, height=bar_h * 0.82, alpha=a, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label_text"].values, fontsize=7.5)
    for ticklbl, color in zip(ax.get_yticklabels(), plot_df["color"].values):
        ticklbl.set_color(color)

    ax.axvline(-np.log10(0.05), color="grey", lw=0.9, ls="--", alpha=0.55)

    ax.set_xlim(0, max_x * 1.12)
    ax.set_ylim(-0.5, n_bars - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        "-log10(q, cross-CT pooled BH, GREAT web server binomial test)\n"
        "label = (intersecting genes / total pathway genes); faded bars = not significant (q>=0.05)",
        fontsize=8.5,
    )
    ax.set_title(
        f"Real GREAT (great.stanford.edu, v3.0.0/hg19) — MSigDB Pathway: REACTOME — "
        f"top {TOP_K_PER_LABEL} per cell type\n"
        f"{describe_tag(tag)}",
        fontsize=10,
    )

    active_labels = plot_df["label"].unique()
    legend_handles = [
        mpatches.Patch(color=CT_COLORS[lbl], label=LABEL_DISPLAY[lbl])
        for lbl in ALL_LABELS if lbl in active_labels
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8,
              loc="lower right", title="Cell type", title_fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG / f"dlpfc_pathway_reactome_GREATweb_top{TOP_K_PER_LABEL}_{tag}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    for tag in ("shaptop2000", "shapdiffpct_top5_top2000"):
        make_figure(tag)
    print("Done.")
