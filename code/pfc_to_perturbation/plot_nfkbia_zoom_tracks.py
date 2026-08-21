#!~/micromamba/envs/sat/bin/python
"""
Genome track (GSE294103 SIRT6 ATAC-seq, group-mean coverage +- SEM per peak)
spanning Peak_50298 (mm10 chr12:55,446,699-55,447,392 -- the peak that lifts
over into the human PFC ATAC-clock's "NFKBIA" feature, hg38
chr14:35,341,586-35,342,086) through the Nfkbia gene body
(chr12:55,489,408-55,492,647), with a gene track below.

Saves: figures/nfkbia_peak50298_to_nfkbia.pdf/.png
"""

import gzip
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrow, Rectangle

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures_v4"
DATA_CSV = ROOT / "data" / "GSE294103_atac_seq_normalized_reads.csv.gz"
SCRATCH = ROOT / "data" / "ucsc_cache"

WIN_CHR = "chr12"
TARGET_START = 55_446_699   # Peak_50298 start
TARGET_END = 55_447_392     # Peak_50298 end

GROUP_ORDER = ["Young WT", "Young SIRT6", "Old WT", "Old SIRT6"]
GROUP_COLORS = {
    "Young WT": "#4393C3",
    "Young SIRT6": "#92C5DE",
    "Old WT": "#D6604D",
    "Old SIRT6": "#762A83",
}


def sample_to_group(col):
    if col.startswith("young_wt"):
        return "Young WT"
    if col.startswith("young_sirt6"):
        return "Young SIRT6"
    if col.startswith("old_wt"):
        return "Old WT"
    if col.startswith("old_sirt6"):
        return "Old SIRT6"
    return "Unknown"


def read_atac_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as raw:
        content = raw.read()
    if content[0] == 0xFF:
        content = content[1:]
    elif content[:3] == b"\xef\xbb\xbf":
        content = content[3:]
    df = pd.read_csv(pd.io.common.StringIO(content.decode("utf-8")))
    df.rename(columns={df.columns[0]: "Chr"}, inplace=True)
    return df


print("Loading GSE294103 peak matrix …")
df = read_atac_csv(DATA_CSV)
sample_cols = [c for c in df.columns if c not in ("Chr", "start_peak", "end_peak", "Peak_ID")]
groups = {c: sample_to_group(c) for c in sample_cols}

df_chr = df[df["Chr"].astype(str) == WIN_CHR.replace("chr", "")].copy()
for g in GROUP_ORDER:
    cols = [c for c in sample_cols if groups[c] == g]
    df_chr[f"mean_{g}"] = df_chr[cols].mean(axis=1)
    df_chr[f"sem_{g}"] = df_chr[cols].sem(axis=1)

with open(SCRATCH / "ucsc_nfkbia_region.json") as f:
    refgene = json.load(f)["refGene"]
by_gene = {}
for rec in refgene:
    sym = rec["name2"]
    if sym not in by_gene or rec["exonCount"] > by_gene[sym]["exonCount"]:
        by_gene[sym] = rec


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Peak_50298's left edge through ~55.50 Mb, Nfkbia marked
# ══════════════════════════════════════════════════════════════════════════════
def plot_to_nfkbia():
    win_start, win_end = TARGET_START, 55_500_000
    win = df_chr[(df_chr["end_peak"] >= win_start) & (df_chr["start_peak"] <= win_end)].copy()
    print(f"  Peaks in window: {len(win)}")

    genes = [g for g in by_gene.values() if g["txEnd"] >= win_start and g["txStart"] <= win_end]
    genes = sorted(genes, key=lambda r: r["txStart"])
    print("  Genes in window:", [g["name2"] for g in genes])

    n_tracks = len(GROUP_ORDER)
    fig, axes = plt.subplots(
        n_tracks + 2, 1, figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6] * n_tracks + [0.35, 1.0], "hspace": 0.12},
    )
    cov_axes = axes[:n_tracks]
    peak_ax = axes[n_tracks]
    gene_ax = axes[n_tracks + 1]

    y_max = max(win[f"mean_{g}"].max() for g in GROUP_ORDER) * 1.15

    for ax, g in zip(cov_axes, GROUP_ORDER):
        col = GROUP_COLORS[g]
        starts = win["start_peak"].values
        ends = win["end_peak"].values
        widths = np.maximum(ends - starts, 250)
        heights = win[f"mean_{g}"].values
        sems = win[f"sem_{g}"].values
        for s, w, h, se in zip(starts, widths, heights, sems):
            ax.add_patch(Rectangle((s, 0), w, h, facecolor=col, edgecolor="none", alpha=0.9))
            ax.plot([s + w / 2, s + w / 2], [h, h + se], color="0.25", linewidth=0.7)
        ax.set_ylim(0, y_max)
        ax.set_ylabel(g, fontsize=9.5, rotation=0, ha="right", va="center", color=col, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.axhline(0, color="0.85", linewidth=0.6)
        ax.axvspan(TARGET_START, TARGET_END, color="#F4A582", alpha=0.35, zorder=0)

    cov_axes[0].text(
        (TARGET_START + TARGET_END) / 2, y_max * 0.93,
        "NFKBIA-linked peak\n(hg38 chr14:35,341,586-35,342,086)",
        ha="left", va="top", fontsize=7.5, color="#B2182B", fontweight="bold",
    )
    axes[0].set_title(
        "GSE294103 SIRT6 ATAC-seq — Peak_50298 → Nfkbia (mm10 chr12)\n"
        f"{WIN_CHR}:{win_start:,}-{win_end:,}   |   mean normalized reads per called peak, ± SEM",
        fontsize=11.5, pad=8,
    )

    # ── Peak track ──
    peak_ax.set_ylim(0, 1)
    peak_ax.set_yticks([])
    for s, e in zip(win["start_peak"], win["end_peak"]):
        w = max(e - s, 250)
        peak_ax.add_patch(Rectangle((s, 0.15), w, 0.7, facecolor="0.35", edgecolor="none"))
    peak_ax.add_patch(Rectangle((TARGET_START, 0.0), TARGET_END - TARGET_START, 1.0,
                                 facecolor="#D6604D", edgecolor="none"))
    peak_ax.set_ylabel("Peaks", fontsize=8.5, rotation=0, ha="right", va="center")
    peak_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # ── Gene track ──
    gene_ax.set_ylim(-0.3, max(len(genes), 1) * 1.1 + 0.3)
    gene_ax.set_yticks([])
    for gi, g in enumerate(genes):
        y = gi * 1.1
        tx_s, tx_e = g["txStart"], g["txEnd"]
        strand = g["strand"]
        gene_ax.plot([tx_s, tx_e], [y, y], color="0.3", linewidth=1.2, zorder=1)

        exon_starts = [int(x) for x in g["exonStarts"].strip(",").split(",")]
        exon_ends = [int(x) for x in g["exonEnds"].strip(",").split(",")]
        for es, ee in zip(exon_starts, exon_ends):
            es_c, ee_c = max(es, win_start), min(ee, win_end)
            if ee_c <= es_c:
                continue
            gene_ax.add_patch(Rectangle((es_c, y - 0.18), ee_c - es_c, 0.36,
                                        facecolor="#4D4D4D", edgecolor="none", zorder=2))

        mid = (max(tx_s, win_start) + min(tx_e, win_end)) / 2
        dx = 2500 if strand == "+" else -2500
        gene_ax.add_patch(FancyArrow(
            mid - dx / 2, y, dx, 0, width=0.001, head_width=0.22, head_length=1100,
            facecolor="0.3", edgecolor="none", zorder=3, length_includes_head=True,
        ))
        is_nfkbia = g["name2"] == "Nfkbia"
        gene_ax.text(
            min(max(tx_s, win_start), win_end), y + 0.35, g["name2"],
            fontsize=9.5 if is_nfkbia else 8.5, style="italic",
            color="#B2182B" if is_nfkbia else "0.15",
            fontweight="bold" if is_nfkbia else "normal",
        )
        if is_nfkbia:
            gene_ax.axvspan(tx_s, tx_e, color="#B2182B", alpha=0.08, zorder=0)

    gene_ax.set_ylabel("Genes", fontsize=8.5, rotation=0, ha="right", va="center")
    gene_ax.spines[["top", "right", "left"]].set_visible(False)
    gene_ax.set_xlim(win_start, win_end)
    gene_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.3f} Mb"))
    gene_ax.set_xlabel(f"mm10 {WIN_CHR} position", fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        out = FIG / f"nfkbia_peak50298_to_nfkbia.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"Saved: {out}")
    plt.close(fig)


plot_to_nfkbia()
