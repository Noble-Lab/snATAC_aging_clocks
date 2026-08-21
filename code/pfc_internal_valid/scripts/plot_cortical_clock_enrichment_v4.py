"""
Plot the PFC cortical-clock enrichment figure: a heatmap and robustness line
across top_n = 100-5000 SHAP-ranked peaks per cell type, a bar chart of
fold-enrichment at top200/10kb, and a table of the top 20 closest
peak-to-clock-CpG pairs at top200/10kb (one row per cell type x peak pair).

Loads: results/cortical_clock_enrichment.csv, results/shap_mean_abs.pkl
Saves: figures/cortical_clock_enrichment_v4.pdf
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from clock_enrichment_lib import annotate_distances, build_hit_list
from plot_clock_enrichment_lib import plot_multilabel_figure, GREATWEB_CT_COLORS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CLOCK_TSV = Path("~/reference_data/dnam_clocks/cortical_clock/cortical_clock_hg38.tsv")
TSS_BED = RESULTS / "great_hg38_tss.bed"
SHAP_PKL = RESULTS / "shap_mean_abs.pkl"

TOP_N_LIST = [100, 200, 500, 1000, 2000, 5000]
LABEL_ORDER = ["All cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


def parse_peak(name):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def main():
    df = pd.read_csv(RESULTS / "cortical_clock_enrichment.csv")

    peak_gene_map = pd.read_csv(RESULTS / "peak_gene_map.csv")
    idx2peakname = dict(zip(peak_gene_map["peak_idx"], peak_gene_map["peak_name"]))
    idx2gene = dict(zip(peak_gene_map["peak_idx"], peak_gene_map["gene_name"]))

    with open(SHAP_PKL, "rb") as f:
        importances = {k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()}
    n = len(importances["All cells"])
    rows = []
    for idx in range(n):
        name = idx2peakname.get(idx)
        if name is None:
            continue
        chrom, start, end = parse_peak(name)
        rows.append((chrom, start, end, idx, idx2gene.get(idx, "?")))
    peaks = pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_idx", "gene_name"])

    cpg_df = pd.read_csv(CLOCK_TSV, sep="\t", header=None,
                          names=["chrom", "start", "end", "strand", "probeID"])
    cpg_df["pos"] = (cpg_df["start"] + cpg_df["end"]) // 2
    tss_df = pd.read_csv(TSS_BED, sep="\t", header=None, names=["chrom", "pos", "strand", "gene"])

    peaks_dist, _, cpg_df2 = annotate_distances(peaks, cpg_df, tss_df)

    idx2gene_by_row = peaks["gene_name"].to_numpy()

    def gene_lookup(row):
        return idx2gene_by_row[row.name] if row.name < len(idx2gene_by_row) else "?"

    hits20 = build_hit_list(peaks_dist, importances, cpg_df2, top_n=200, window=10_000,
                             cpg_id_col="probeID", gene_lookup=gene_lookup, top_k=20, dedup_peaks=False)

    plot_multilabel_figure(
        df, hits20, LABEL_ORDER, TOP_N_LIST, primary_window=10_000, primary_top_n=200,
        out_path_stub=ROOT / "figures" / "cortical_clock_enrichment_v4",
        suptitle="PFC internal age clock: overlap with the Shireby cortical methylation clock",
        subtitle_a="PFC age-predictive peaks vs. cortical-clock CpGs (±10kb)",
        subtitle_b="top200 peaks, within 10kb of a clock CpG",
        table_title="Top 20 closest peak-to-clock-CpG pairs (top200 @10kb, one row per cell type)",
        colors=GREATWEB_CT_COLORS,
    )


if __name__ == "__main__":
    main()
