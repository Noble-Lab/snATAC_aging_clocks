"""
Shared plotting for TSS-stratified clock-enrichment results (see
clock_enrichment_lib.py). plot_multilabel_figure() draws a 5-panel figure for
datasets with a full label x top_n x window grid (PFC, hippocampus, mouse
hypothalamus): a heatmap and a robustness-across-thresholds line (both
BH-corrected across cell types per threshold), a bar chart at a
representative threshold, and a table of the top peak/CpG pairs.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from clock_enrichment_lib import add_bh_qvalues

mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

DEFAULT_CT_COLORS = {
    "All cells": "#444444", "All_cells": "#444444",
    "Excitatory": "#E64B35", "Inhibitory": "#4DBBD5",
    "Oligo": "#00A087", "Astro": "#3C5488", "Microglia": "#F39B7F", "OPC": "#8491B4",
}

# Matches pfc_internal_valid/scripts/great_web_top5.py's CT_COLORS exactly
# (used for pathway_enrichment_GREATweb_MSigDBPathway_top5_*.pdf), so the
# per-cell-type figures share a color language with the pathway-enrichment
# figures from the same project.
GREATWEB_CT_COLORS = {
    "All cells": "black", "All_cells": "black",
    "Excitatory": "tab:orange", "Inhibitory": "tab:green",
    "Oligo": "tab:blue", "Astro": "tab:red",
    "Microglia": "tab:purple", "OPC": "tab:brown",
}


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_multilabel_figure(df, hits20, label_order, top_n_list, primary_window, primary_top_n,
                            out_path_stub, suptitle, subtitle_a, subtitle_b, table_title,
                            colors=None, label_col="label"):
    """
    Draws: (A) heatmap of z-score vs TSS-matched null across label x top_n,
    stars = BH q (corrected across cell types per threshold); (B) bar chart
    of fold-enrichment at (primary_window, primary_top_n); (C) fold-enrichment
    vs top_n line per cell type, with q-value stars stacked above each point;
    (D) a table of the top peak/CpG pairs (hits20).
    """
    colors = colors or DEFAULT_CT_COLORS
    df = add_bh_qvalues(df, group_cols=["top_n", "window"], p_col="perm_p", q_col="q")

    fig = plt.figure(figsize=(13, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.85, 1.7], width_ratios=[1.3, 1],
                           hspace=0.5, wspace=0.32)

    # --- Panel A: heatmap (q-value stars) ---
    axA = fig.add_subplot(gs[0, 0])
    sub = df[df["window"] == primary_window]
    mat = sub.pivot(index=label_col, columns="top_n", values="z").reindex(
        index=label_order, columns=top_n_list)
    qmat = sub.pivot(index=label_col, columns="top_n", values="q").reindex(
        index=label_order, columns=top_n_list)
    vmax = np.nanmax(np.abs(mat.to_numpy())) if np.isfinite(mat.to_numpy()).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = axA.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axA.set_xticks(range(len(top_n_list)))
    axA.set_xticklabels([f"top{n}" for n in top_n_list], fontsize=8.5)
    axA.set_yticks(range(len(label_order)))
    axA.set_yticklabels(label_order, fontsize=9)
    for i, label in enumerate(label_order):
        for j, n in enumerate(top_n_list):
            z, q = mat.iloc[i, j], qmat.iloc[i, j]
            if np.isnan(z):
                continue
            axA.text(j, i, f"{z:.1f}{stars(q)}", ha="center", va="center", fontsize=7.2,
                      color="white" if abs(z) > vmax * 0.55 else "black")
    cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cbar.set_label("z-score vs TSS-matched null", fontsize=8.5)
    axA.set_title(subtitle_a + "\n(stars = BH q, corrected across cell types per threshold)",
                  fontsize=9.5)

    # --- Panel B: bar at representative threshold (q-value stars) ---
    axB = fig.add_subplot(gs[0, 1])
    sub2 = df[(df["window"] == primary_window) & (df["top_n"] == primary_top_n)]
    sub2 = sub2.set_index(label_col).reindex(label_order)
    bar_colors = [colors.get(l, "#888888") for l in label_order]
    bars = axB.bar(label_order, sub2["fold_enrichment"], color=bar_colors, edgecolor="none")
    axB.axhline(1, color="#999999", lw=1, ls="--", zorder=0)
    axB.set_ylabel("Fold enrichment vs TSS-matched null", fontsize=9)
    axB.set_title(subtitle_b, fontsize=10)
    axB.set_xticks(range(len(label_order)))
    axB.set_xticklabels(label_order, rotation=40, ha="right", fontsize=8.5)
    max_h = np.nanmax(sub2["fold_enrichment"].to_numpy()) if len(sub2) else 1.0
    max_h = max_h if np.isfinite(max_h) else 1.0
    axB.set_ylim(top=max_h * 1.22)
    for bar, (_, row) in zip(bars, sub2.iterrows()):
        h = bar.get_height() if np.isfinite(bar.get_height()) else 0
        axB.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.03, stars(row["q"]),
                  ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Panel C: robustness line, square-ish, with stacked significance stars ---
    # Square box (not full-width) so it doesn't dominate the figure; stars for
    # ALL significant labels at a given top-N are stacked in a column above the
    # tallest line at that x, in a fixed label order, so they never overlap
    # each other or any line.
    axC = fig.add_subplot(gs[1, 0])
    axC.set_box_aspect(1)
    label_lines = {}
    y_top_by_topn = {n: -np.inf for n in top_n_list}
    for label in label_order:
        s = df[(df[label_col] == label) & (df["window"] == primary_window)].sort_values("top_n")
        if len(s) == 0:
            continue
        color = colors.get(label, "#888888")
        axC.plot(s["top_n"], s["fold_enrichment"], marker="o", ms=4, lw=1.6,
                  color=color, label=label)
        label_lines[label] = s.set_index("top_n")
        for n, row in s.set_index("top_n").iterrows():
            if np.isfinite(row["fold_enrichment"]):
                y_top_by_topn[n] = max(y_top_by_topn[n], row["fold_enrichment"])

    global_max = max([v for v in y_top_by_topn.values() if np.isfinite(v)] + [1.0])
    step = 0.09 * global_max
    max_stack = 0
    for n in top_n_list:
        if not np.isfinite(y_top_by_topn[n]):
            continue
        k = 0
        for label in label_order:
            s = label_lines.get(label)
            if s is None or n not in s.index:
                continue
            row = s.loc[n]
            st = stars(row["q"])
            if not st:
                continue
            y = y_top_by_topn[n] + (k + 1) * step
            axC.text(n, y, st, ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                      color=colors.get(label, "#888888"))
            k += 1
        max_stack = max(max_stack, k)

    axC.axhline(1, color="#999999", lw=1, ls="--", zorder=0)
    axC.set_xscale("log")
    axC.set_xticks(top_n_list)
    axC.set_xticklabels([str(n) for n in top_n_list], fontsize=8, rotation=30)
    axC.set_xlabel("top-N age-predictive features", fontsize=9.5)
    axC.set_ylabel(f"Fold enrichment (±{primary_window // 1000}kb)", fontsize=9.5)
    axC.set_ylim(top=global_max + (max_stack + 1.5) * step)
    axC.set_title("Robustness across rank thresholds\n(stars = BH q<0.05/0.01/0.001, cross-CT\n"
                  "corrected, stacked above highest line)", fontsize=9.5)

    axLeg = fig.add_subplot(gs[1, 1])
    axLeg.axis("off")
    handles, labels_ = axC.get_legend_handles_labels()
    axLeg.legend(handles, labels_, fontsize=9.5, frameon=False, loc="center left",
                 title="Cell type", title_fontsize=10)

    # --- Panel D: full-width, 20-row hit table with peak + CpG coordinate columns ---
    axD = fig.add_subplot(gs[2, :])
    axD.axis("off")
    if len(hits20) == 0:
        axD.text(0.5, 0.5, "No peaks within window at this threshold", ha="center",
                  va="center", fontsize=9, transform=axD.transAxes, color="#666666")
    else:
        has_gene = "gene" in hits20.columns
        rows = []
        for r in hits20.itertuples():
            peak_str = f"{r.chrom}:{r.start}-{r.end}"
            cpg_str = f"{r.cpg_chrom}:{r.cpg_pos}"
            row = [r.label, peak_str]
            if has_gene:
                row.append(r.gene)
            row += [cpg_str, r.cpg_id, f"{int(r.dist_cpg)} bp"]
            rows.append(row)
        col_labels = ["cell type(s)", "ATAC peak"]
        col_widths = [0.20, 0.18]
        if has_gene:
            col_labels.append("gene")
            col_widths.append(0.09)
        col_labels += ["CpG site", "clock locus ID", "distance"]
        col_widths += [0.18, 0.23, 0.12]
        tbl = axD.table(cellText=rows, colLabels=col_labels, cellLoc="left", colLoc="left",
                         bbox=[0, 0, 1, 0.95], colWidths=col_widths)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_text_props(fontweight="bold")
    axD.set_title(table_title, fontsize=10.5, pad=6)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.998)
    out_path_stub = Path(out_path_stub)
    out_path_stub.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(out_path_stub) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path_stub) + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_path_stub}.pdf")
