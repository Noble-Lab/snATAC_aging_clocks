"""
Plot the mouse hypothalamus clock enrichment figure: for each cell type,
fold-enrichment of top age-predictive peaks near the Meer et al. 2018 mouse
whole-lifespan methylation clock CpGs across top_n thresholds, plus a table
of the top 20 closest peak-to-clock-CpG pairs at top200/10kb (one row per
cell type x peak pair).

Loads: results/mouse_clock_enrichment.csv, results/peak_importance.pkl
Saves: figures/mouse_clock_enrichment_v4.pdf
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pfc_internal_valid" / "scripts"))
from clock_enrichment_lib import annotate_distances, build_hit_list
from plot_clock_enrichment_lib import plot_multilabel_figure, GREATWEB_CT_COLORS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MOUSE_CLOCK_HG38 = Path("~/reference_data/dnam_clocks/mouse_clock/mouse_wholelifespan_clock_hg38.tsv")
TSS_BED = ROOT.parent / "pfc_internal_valid" / "results" / "great_hg38_tss.bed"
PEAK_GENE_MAP = ROOT.parent / "pfc_internal_valid" / "results" / "peak_gene_map.csv"

TOP_N_LIST = [100, 200, 500, 1000, 2000, 5000]
LABEL_ORDER = ["All_cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


def parse_peak(name):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def main():
    df = pd.read_csv(RESULTS / "mouse_clock_enrichment.csv")

    with open(RESULTS / "peak_importance.pkl", "rb") as f:
        d = pickle.load(f)
    peak_names = d["All_cells"]["peak_names"]
    peaks = pd.DataFrame([parse_peak(n) for n in peak_names], columns=["chrom", "start", "end"])
    importances = {label: v["importance"] for label, v in d.items()}

    clock = pd.read_csv(MOUSE_CLOCK_HG38, sep="\t")
    cpg_df = pd.DataFrame({"chrom": clock["chrom"], "pos": clock["pos"].astype(int),
                            "gene_mm10": clock["gene_mm10"]})
    tss_df = pd.read_csv(TSS_BED, sep="\t", header=None, names=["chrom", "pos", "strand", "gene"])
    peaks_dist, _, cpg_df2 = annotate_distances(peaks, cpg_df, tss_df)
    cpg_df2["cpg_label"] = "mm10 gene: " + cpg_df2["gene_mm10"].fillna("-")

    gene_map = pd.read_csv(PEAK_GENE_MAP)
    gene_lookup_dict = dict(zip(gene_map["peak_name"], gene_map["gene_name"]))

    def gene_lookup(row):
        return gene_lookup_dict.get(f"{row['chrom']}:{row['start']}-{row['end']}", "?")

    hits20 = build_hit_list(peaks_dist, importances, cpg_df2, top_n=200, window=10_000,
                             cpg_id_col="cpg_label", gene_lookup=gene_lookup, top_k=20, dedup_peaks=False)

    plot_multilabel_figure(
        df, hits20, LABEL_ORDER, TOP_N_LIST, primary_window=10_000, primary_top_n=200,
        out_path_stub=ROOT / "figures" / "mouse_clock_enrichment_v4",
        suptitle="Mouse hypothalamus: overlap with the Meer mouse whole-lifespan DNAm clock",
        subtitle_a="Hypothalamus age-predictive peaks (hg38) vs. mouse-clock CpGs (mm10→hg38 lifted, ±10kb)",
        subtitle_b="top200 peaks, within 10kb of a lifted clock CpG",
        table_title="Top 20 closest peak-to-clock-CpG pairs (top200 @10kb, one row per cell type)",
        colors=GREATWEB_CT_COLORS,
    )


if __name__ == "__main__":
    main()
