"""
Export top-N SHAP-ranked ATAC peaks (hg38, per cell type / all cells) as BED
files, for submission to the real GREAT web server (great.stanford.edu) via
rGREAT's submitGreatJob() in scripts/great_web_sweep.R.

Loads: results/shap_mean_abs.pkl, results/peak_gene_map.csv
Saves: results/great_web_beds/{label}_top{N}.bed   (chr, start, end; 0-based half-open)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BED_DIR = RESULTS / "great_web_beds"
BED_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPES = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
ALL_LABELS = ["All cells"] + CELL_TYPES
TOP_N_LIST = [2000]  # top-2000-SHAP-ranked peaks per label/cell type

with open(RESULTS / "shap_mean_abs.pkl", "rb") as f:
    shap_results = pickle.load(f)
peak_gene_map = pd.read_csv(RESULTS / "peak_gene_map.csv")
idx2peakname = dict(zip(peak_gene_map["peak_idx"], peak_gene_map["peak_name"]))


def parse_peak(name):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


for top_n in TOP_N_LIST:
    for label in ALL_LABELS:
        s = shap_results[label]
        sorted_idx = np.argsort(s)[::-1][:top_n]
        rows = [parse_peak(idx2peakname[int(idx)]) for idx in sorted_idx if int(idx) in idx2peakname]
        bed_path = BED_DIR / f"{label.replace(' ', '_')}_top{top_n}.bed"
        with open(bed_path, "w") as f:
            for chrom, start, end in rows:
                f.write(f"{chrom}\t{start}\t{end}\n")
        print(f"{label} top{top_n}: {len(rows)} peaks -> {bed_path}")

print("Done.")
