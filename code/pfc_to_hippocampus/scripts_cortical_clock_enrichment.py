"""
Human hippocampus: are top age-predictive PFC->hippocampus peaks (hg38, from
peak_importance.pkl) enriched near the Shireby et al. 2020 cortical clock CpGs?
Same coordinate space as PFC (hg38) -- no liftover needed, direct reuse of the
PFC internal clock's external CpG set and TSS annotation.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pfc_internal_valid" / "scripts"))
from clock_enrichment_lib import run_enrichment

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CLOCK_TSV = Path("~/reference_data/dnam_clocks/cortical_clock/cortical_clock_hg38.tsv")
TSS_BED = ROOT.parent / "pfc_internal_valid" / "results" / "great_hg38_tss.bed"

TOP_N_LIST = [100, 200, 500, 1000, 2000, 5000]
WINDOWS = [10_000]  # bp, distance defining "near" a clock CpG


def parse_peak(name):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def main():
    with open(RESULTS / "peak_importance.pkl", "rb") as f:
        d = pickle.load(f)

    ref_label = "All_cells"
    peak_names = d[ref_label]["peak_names"]
    parsed = [parse_peak(n) for n in peak_names]
    peaks = pd.DataFrame(parsed, columns=["chrom", "start", "end"])
    print(f"Background: {len(peaks)} peaks (hg38)")

    importances = {}
    for label, v in d.items():
        assert len(v["peak_names"]) == len(peak_names), f"{label} peak set differs from {ref_label}"
        assert np.array_equal(v["peak_names"], peak_names), f"{label} peak order differs from {ref_label}"
        importances[label] = v["importance"]

    cpg_df = pd.read_csv(CLOCK_TSV, sep="\t", header=None,
                          names=["chrom", "start", "end", "strand", "probeID"])
    cpg_df["pos"] = (cpg_df["start"] + cpg_df["end"]) // 2
    tss_df = pd.read_csv(TSS_BED, sep="\t", header=None, names=["chrom", "pos", "strand", "gene"])

    out, peaks_dist = run_enrichment(peaks, importances, cpg_df, tss_df, TOP_N_LIST, WINDOWS)
    out.to_csv(RESULTS / "cortical_clock_enrichment.csv", index=False)
    print(out[out["window"] == 10_000].to_string(index=False))
    print(f"\nSaved {RESULTS / 'cortical_clock_enrichment.csv'}")

    for _, r in out.iterrows():
        log_test(
            "pfc_to_hippocampus", "cortical_clock_enrichment_v4",
            analysis="TSS-stratified permutation enrichment vs published DNAm-clock CpGs",
            test="stratified_permutation_test", group_a=r.get("label"),
            group_b=f"top{r.get('top_n')}_within{r.get('window')}bp",
            n_a=r.get("top_n"), statistic=r.get("z"), effect_size=r.get("fold_enrichment"),
            p_value=r.get("perm_p"), notes="pfc_to_hippocampus all_pfc Ridge clock",
        )


if __name__ == "__main__":
    main()
