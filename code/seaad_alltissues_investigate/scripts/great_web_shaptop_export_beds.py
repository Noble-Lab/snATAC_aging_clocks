"""Export top-N SHAP-ranked peaks per label (hg38) as BED files, for
submission to the REAL GREAT web server (great.stanford.edu) via
rGREAT's submitGreatJob() in scripts/great_web_dlpfc_sweep.R.

Peak sets = top-N peaks by mean |SHAP| across all DLPFC donors, from
shap_dlpfc_analysis.py's cache/shap_adnc_cache.pkl, exported as raw genomic
regions instead of being collapsed to nearest-TSS gene symbols first --
GREAT assigns its own regulatory-domain-based gene mapping.

Loads: cache/shap_adnc_cache.pkl
Saves: results/great_web_dlpfc_beds/{label}__shaptop{N}.bed
       (skips a BED that already exists, so re-running only fills in new N)
"""
import pickle
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
BED_DIR = ROOT / "results" / "great_web_dlpfc_beds"
BED_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_PEAKS_LIST = [2000]  # top-2000-SHAP-ranked peaks per label


def parse_peak(name: str):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def main():
    cache_path = ROOT / "cache" / "shap_adnc_cache.pkl"
    print(f"Loading SHAP cache ({cache_path}) ...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    shap_results = cache["shap_results"]
    feat_names   = cache["feat_names"]

    for label in cache["ALL_LABELS"]:
        shap = shap_results[label]
        for top_n_peaks in TOP_N_PEAKS_LIST:
            bed_path = BED_DIR / f"{label}__shaptop{top_n_peaks}.bed"
            if bed_path.exists():
                print(f"  [{label}] shaptop{top_n_peaks}: already exists, skipping")
                continue
            top_idx = np.argsort(shap)[::-1][:top_n_peaks]
            with open(bed_path, "w") as f:
                for i in top_idx:
                    chrom, start, end = parse_peak(feat_names[i])
                    f.write(f"{chrom}\t{start}\t{end}\n")
            print(f"  [{label}] {len(top_idx)} peaks -> {bed_path}")

    print("Done.")


if __name__ == "__main__":
    main()
