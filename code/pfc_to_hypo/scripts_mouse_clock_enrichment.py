"""
Mouse hypothalamus: are top age-predictive PFC->hypothalamus peaks (hg38,
from peak_importance.pkl) enriched near the Meer et al. 2018 mouse
whole-lifespan multi-tissue DNAm clock CpGs, lifted mm10->hg38?

The hypothalamus clock's shared-feature space is expressed in PFC's own
hg38 peak coordinates (via liftover-based intersection at data-prep time),
so rather than lifting the peaks to mm10, we lift the (smaller) mouse clock
CpG set to hg38 and run the whole test in hg38 space, reusing the PFC
internal clock's TSS annotation.
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
MOUSE_CLOCK_HG38 = Path("~/reference_data/dnam_clocks/mouse_clock/mouse_wholelifespan_clock_hg38.tsv")
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
    print(f"Background: {len(peaks)} peaks (hg38, shared PFC<->mouse-hypothalamus)")

    importances = {}
    for label, v in d.items():
        assert np.array_equal(v["peak_names"], peak_names), f"{label} peak order differs"
        importances[label] = v["importance"]

    clock = pd.read_csv(MOUSE_CLOCK_HG38, sep="\t")
    cpg_df = pd.DataFrame({"chrom": clock["chrom"], "pos": clock["pos"].astype(int)})
    print(f"Mouse clock CpGs lifted to hg38: {len(cpg_df)}")

    tss_df = pd.read_csv(TSS_BED, sep="\t", header=None, names=["chrom", "pos", "strand", "gene"])

    out, peaks_dist = run_enrichment(peaks, importances, cpg_df, tss_df, TOP_N_LIST, WINDOWS)
    out.to_csv(RESULTS / "mouse_clock_enrichment.csv", index=False)
    print(out[out["window"] == 10_000].to_string(index=False))
    print(f"\nSaved {RESULTS / 'mouse_clock_enrichment.csv'}")

    for _, r in out.iterrows():
        log_test(
            "pfc_to_hypo", "mouse_clock_enrichment_v4",
            analysis="TSS-stratified permutation enrichment vs Meer et al. 2018 mouse DNAm clock",
            test="stratified_permutation_test", group_a=r.get("label"),
            group_b=f"top{r.get('top_n')}_within{r.get('window')}bp",
            n_a=r.get("top_n"), statistic=r.get("z"), effect_size=r.get("fold_enrichment"),
            p_value=r.get("perm_p"), notes="pfc_to_hypo peak_importance.pkl clock",
        )


if __name__ == "__main__":
    main()
