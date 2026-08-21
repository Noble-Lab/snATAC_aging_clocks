"""
Test whether PFC ATAC peaks that are top age predictors (SHAP-ranked) sit
closer to the 347 CpGs of the Shireby et al. 2020 cortical epigenetic clock
(Brain 143:3763) than expected by chance.

Null model is stratified by distance-to-nearest-TSS (10 quantile bins of the
full 521k-peak background), so the test controls for "top peaks are just
promoter-proximal" and asks specifically about proximity to clock CpGs beyond
that. Per-bin null counts are drawn from a hypergeometric distribution
(exact for sampling without replacement within a bin) and summed across bins.

Loads: results/shap_mean_abs.pkl (full background, hg38 500bp peaks),
       results/peak_gene_map.csv,
       results/great_hg38_tss.bed,
       <tmpdir>/cortical_clock_hg38.tsv (347 CpGs, hg38, from EPIC manifest)
Saves: results/cortical_clock_enrichment.csv, figures/cortical_clock_enrichment.pdf
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
CLOCK_TSV = Path("~/reference_data/dnam_clocks/cortical_clock/cortical_clock_hg38.tsv")
TSS_BED = RESULTS / "great_hg38_tss.bed"

WINDOWS = [10_000]  # bp, distance defining "near" a clock CpG
TOP_N_LIST = [100, 200, 500, 1000, 2000, 5000]
N_TSS_BINS = 10
N_PERM = 20_000
RNG = np.random.default_rng(0)


def parse_peak(name):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def load_clock_cpgs():
    df = pd.read_csv(CLOCK_TSV, sep="\t", header=None,
                      names=["chrom", "start", "end", "strand", "probeID"])
    df["mid"] = (df["start"] + df["end"]) // 2
    return df


def load_tss():
    df = pd.read_csv(TSS_BED, sep="\t", header=None,
                      names=["chrom", "pos", "strand", "gene"])
    return df


def load_peaks():
    with open(RESULTS / "shap_mean_abs.pkl", "rb") as f:
        shap_results = {k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()}

    peak_gene_map = pd.read_csv(RESULTS / "peak_gene_map.csv")
    idx2peakname = dict(zip(peak_gene_map["peak_idx"], peak_gene_map["peak_name"]))
    idx2gene = dict(zip(peak_gene_map["peak_idx"], peak_gene_map["gene_name"]))
    n = len(next(iter(shap_results.values())))
    rows = []
    for idx in range(n):
        name = idx2peakname.get(idx)
        if name is None:
            continue
        chrom, start, end = parse_peak(name)
        rows.append((chrom, start, end, idx, idx2gene.get(idx, "?")))
    peaks = pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_idx", "gene_name"])
    peaks["mid"] = (peaks["start"] + peaks["end"]) // 2
    return peaks, shap_results


def build_pos_lookup(df, pos_col="mid"):
    lookup = {}
    for chrom, sub in df.groupby("chrom"):
        lookup[chrom] = np.sort(sub[pos_col].to_numpy())
    return lookup


def min_dist_vectorized(peaks, lookup):
    out = np.full(len(peaks), np.inf)
    for chrom, sub in peaks.groupby("chrom"):
        arr = lookup.get(chrom)
        if arr is None or len(arr) == 0:
            continue
        pos = sub["mid"].to_numpy()
        i = np.searchsorted(arr, pos)
        d = np.full(len(pos), np.inf)
        right = i < len(arr)
        d[right] = np.minimum(d[right], np.abs(arr[np.clip(i[right], 0, len(arr) - 1)] - pos[right]))
        left = i > 0
        d[left] = np.minimum(d[left], np.abs(arr[np.clip(i[left] - 1, 0, len(arr) - 1)] - pos[left]))
        out[sub.index.to_numpy()] = d
    return out


def nearest_probe_ids(peaks, cpg_df):
    """probeID of the nearest clock CpG for every peak (parallel to dist_cpg)."""
    out = np.full(len(peaks), "", dtype=object)
    for chrom, sub_cpg in cpg_df.groupby("chrom"):
        order = np.argsort(sub_cpg["mid"].to_numpy())
        mids = sub_cpg["mid"].to_numpy()[order]
        probes = sub_cpg["probeID"].to_numpy()[order]
        sub_peaks = peaks[peaks["chrom"] == chrom]
        if len(sub_peaks) == 0:
            continue
        pos = sub_peaks["mid"].to_numpy()
        i = np.searchsorted(mids, pos)
        i_right = np.clip(i, 0, len(mids) - 1)
        i_left = np.clip(i - 1, 0, len(mids) - 1)
        d_right = np.abs(mids[i_right] - pos)
        d_left = np.abs(mids[i_left] - pos)
        use_right = d_right <= d_left
        nearest = np.where(use_right, probes[i_right], probes[i_left])
        out[sub_peaks.index.to_numpy()] = nearest
    return out


def stratified_null(is_top, within, tss_bin, n_bins, n_perm, rng):
    """Hypergeometric null: for each TSS-distance bin, draw as many peaks as
    the real top set drew from that bin, without replacement, and count how
    many land within WINDOW of a clock CpG. Sum across bins per permutation."""
    null_counts = np.zeros(n_perm, dtype=np.int64)
    for b in range(n_bins):
        in_bin = tss_bin == b
        N_b = int(in_bin.sum())
        if N_b == 0:
            continue
        W_b = int((in_bin & within).sum())
        n_b = int((in_bin & is_top).sum())
        if n_b == 0:
            continue
        draws = rng.hypergeometric(ngood=W_b, nbad=N_b - W_b, nsample=n_b, size=n_perm)
        null_counts += draws
    return null_counts


def main():
    print("Loading clock CpGs, TSS annotations, and PFC peaks...")
    cpg_df = load_clock_cpgs()
    tss_df = load_tss()
    peaks, shap_results = load_peaks()
    print(f"  {len(cpg_df)} clock CpGs, {len(tss_df)} TSS, {len(peaks)} background peaks")

    cpg_lookup = build_pos_lookup(cpg_df)
    tss_lookup = build_pos_lookup(tss_df.rename(columns={"pos": "mid"}))

    peaks["dist_cpg"] = min_dist_vectorized(peaks, cpg_lookup)
    peaks["dist_tss"] = min_dist_vectorized(peaks, tss_lookup)

    # global TSS-distance bins (quantiles of the full background), reused across labels/top_n
    edges = np.quantile(peaks["dist_tss"], np.linspace(0, 1, N_TSS_BINS + 1))
    edges[0], edges[-1] = -1, np.inf
    tss_bin = np.digitize(peaks["dist_tss"].to_numpy(), edges[1:-1])

    rows = []
    for label, shap_vec in shap_results.items():
        order = np.argsort(shap_vec)[::-1]
        print(f"\n=== {label} ===")
        for top_n in TOP_N_LIST:
            top_peak_ids = set(order[:top_n].tolist())
            is_top = peaks["peak_idx"].isin(top_peak_ids).to_numpy()

            for window in WINDOWS:
                within = (peaks["dist_cpg"] <= window).to_numpy()
                n_obs = int((is_top & within).sum())
                null_counts = stratified_null(is_top, within, tss_bin, N_TSS_BINS, N_PERM, RNG)
                null_mean, null_std = null_counts.mean(), null_counts.std()
                z = (n_obs - null_mean) / null_std if null_std > 0 else np.nan
                pval = (np.sum(null_counts >= n_obs) + 1) / (N_PERM + 1)
                fold = n_obs / null_mean if null_mean > 0 else np.nan

                rows.append(dict(label=label, top_n=top_n, window=window,
                                  n_within=n_obs, frac_within=n_obs / top_n,
                                  null_mean=null_mean, null_std=null_std,
                                  fold_enrichment=fold, z=z, perm_p=pval))
                log_test(
                    "pfc_internal_valid", "cortical_clock_enrichment_v4",
                    analysis="TSS-stratified permutation enrichment vs published DNAm-clock CpGs",
                    test="stratified_permutation_test", group_a=label,
                    group_b=f"top{top_n}_within{window}bp", n_a=top_n,
                    statistic=z, effect_size=fold, p_value=pval,
                    notes=f"n_perm={N_PERM}; n_within={n_obs}",
                )
                print(f"  top{top_n} @{window//1000}kb: {n_obs}/{top_n} within window "
                      f"(TSS-matched null={null_mean:.1f}±{null_std:.1f}, "
                      f"fold={fold:.2f}, z={z:.2f}, p={pval:.4f})")

    out = pd.DataFrame(rows)
    out_path = RESULTS / "cortical_clock_enrichment.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    peaks[["peak_idx", "chrom", "start", "end", "gene_name", "dist_cpg", "dist_tss"]].to_csv(
        RESULTS / "cortical_clock_peak_distances.csv", index=False)

    # Annotated overlap list: which specific top peaks/genes drive the signal,
    # at a representative threshold (top2000 @10kb).
    nearest_probe = nearest_probe_ids(peaks, cpg_df)
    hit_rows = []
    for label, shap_vec in shap_results.items():
        order = np.argsort(shap_vec)[::-1][:2000]
        rank = {int(pi): r for r, pi in enumerate(order)}
        sub = peaks[peaks["peak_idx"].isin(rank) & (peaks["dist_cpg"] <= 10_000)].copy()
        sub["label"] = label
        sub["rank"] = sub["peak_idx"].map(rank)
        sub["probeID"] = nearest_probe[sub.index.to_numpy()]
        hit_rows.append(sub)
    hits = pd.concat(hit_rows, ignore_index=True)
    hits = hits[["label", "rank", "chrom", "start", "end", "gene_name", "probeID", "dist_cpg"]]
    hits = hits.sort_values(["label", "rank"])
    hits_path = RESULTS / "cortical_clock_overlap_hits_top2000_10kb.csv"
    hits.to_csv(hits_path, index=False)
    print(f"Saved {hits_path} ({len(hits)} peak-CpG hits across {hits['label'].nunique()} labels)")


if __name__ == "__main__":
    main()
