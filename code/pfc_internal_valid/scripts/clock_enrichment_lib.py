"""
Shared TSS-stratified enrichment-test machinery: tests whether the top-N
importance-ranked peaks in a (background peaks, importance scores, clock
CpGs, TSS annotation) tuple sit closer to a set of external "clock" loci than
a null where peaks are drawn with the same per-bin distance-to-nearest-TSS
composition as the real top-N set (removes the "top peaks / clock loci are
both just promoter-proximal" confound). Genome-build/species-agnostic; used
by cortical_clock_enrichment.py, scripts_cortical_clock_enrichment.py, and
scripts_mouse_clock_enrichment.py.
"""
import numpy as np
import pandas as pd

N_TSS_BINS = 10


def build_pos_lookup(df, chrom_col="chrom", pos_col="mid"):
    lookup = {}
    for chrom, sub in df.groupby(chrom_col):
        lookup[chrom] = np.sort(sub[pos_col].to_numpy())
    return lookup


def min_dist_vectorized(peaks, lookup, chrom_col="chrom", pos_col="mid"):
    out = np.full(len(peaks), np.inf)
    for chrom, sub in peaks.groupby(chrom_col):
        arr = lookup.get(chrom)
        if arr is None or len(arr) == 0:
            continue
        pos = sub[pos_col].to_numpy()
        i = np.searchsorted(arr, pos)
        d = np.full(len(pos), np.inf)
        right = i < len(arr)
        d[right] = np.minimum(d[right], np.abs(arr[np.clip(i[right], 0, len(arr) - 1)] - pos[right]))
        left = i > 0
        d[left] = np.minimum(d[left], np.abs(arr[np.clip(i[left] - 1, 0, len(arr) - 1)] - pos[left]))
        out[sub.index.to_numpy()] = d
    return out


def stratified_null(is_top, within, tss_bin, n_bins, n_perm, rng):
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


def annotate_distances(peaks, cpg_df, tss_df):
    """Compute dist_cpg/dist_tss (+ TSS quantile bin) for every peak. This is the
    expensive-permutation-free prefix of run_enrichment -- reuse it standalone
    when you already have enrichment stats cached and just need peak-CpG
    distances (e.g. to rebuild a hit table without re-running permutations)."""
    peaks = peaks.copy()
    peaks["mid"] = (peaks["start"] + peaks["end"]) // 2

    cpg_df = cpg_df.copy()
    if "mid" not in cpg_df.columns:
        cpg_df["mid"] = cpg_df["pos"]
    tss_df = tss_df.copy()
    if "mid" not in tss_df.columns:
        tss_df["mid"] = tss_df["pos"]

    cpg_lookup = build_pos_lookup(cpg_df)
    tss_lookup = build_pos_lookup(tss_df)

    peaks["dist_cpg"] = min_dist_vectorized(peaks, cpg_lookup)
    peaks["dist_tss"] = min_dist_vectorized(peaks, tss_lookup)

    edges = np.quantile(peaks["dist_tss"], np.linspace(0, 1, N_TSS_BINS + 1))
    edges[0], edges[-1] = -1, np.inf
    tss_bin = np.digitize(peaks["dist_tss"].to_numpy(), edges[1:-1])
    return peaks, tss_bin, cpg_df


def run_enrichment(peaks, importances, cpg_df, tss_df, top_n_list, windows,
                    n_perm=20_000, seed=0, label_col="label"):
    """
    peaks:       DataFrame[chrom, start, end] -- one row per background peak/tile
    importances: dict[label -> np.array of len(peaks)], higher = more age-important
    cpg_df:      DataFrame[chrom, pos] -- external clock loci (already on the same
                 genome build as `peaks`)
    tss_df:      DataFrame[chrom, pos] -- TSS annotation on the same genome build
    Returns: (results_df, peaks_with_dist)
    """
    rng = np.random.default_rng(seed)
    peaks, tss_bin, cpg_df = annotate_distances(peaks, cpg_df, tss_df)

    rows = []
    for label, imp in importances.items():
        order = np.argsort(imp)[::-1]
        for top_n in top_n_list:
            if top_n > len(peaks):
                continue
            top_idx = order[:top_n]
            is_top = np.zeros(len(peaks), dtype=bool)
            is_top[top_idx] = True
            for window in windows:
                within = (peaks["dist_cpg"] <= window).to_numpy()
                n_obs = int((is_top & within).sum())
                null_counts = stratified_null(is_top, within, tss_bin, N_TSS_BINS, n_perm, rng)
                null_mean, null_std = null_counts.mean(), null_counts.std()
                z = (n_obs - null_mean) / null_std if null_std > 0 else np.nan
                pval = (np.sum(null_counts >= n_obs) + 1) / (n_perm + 1)
                fold = n_obs / null_mean if null_mean > 0 else np.nan
                rows.append({label_col: label, "top_n": top_n, "window": window,
                             "n_within": n_obs, "frac_within": n_obs / top_n,
                             "null_mean": null_mean, "null_std": null_std,
                             "fold_enrichment": fold, "z": z, "perm_p": pval})
    return pd.DataFrame(rows), peaks


def nearest_cpg_info(peaks, cpg_df, id_col=None):
    """Nearest clock CpG's (chrom, pos, id) for every peak (parallel arrays to dist_cpg).
    cpg_df must already have a 'mid' column (as built inside run_enrichment)."""
    id_col = id_col or cpg_df.columns[0]
    n = len(peaks)
    out_chrom = np.full(n, "", dtype=object)
    out_pos = np.full(n, -1, dtype=np.int64)
    out_id = np.full(n, "", dtype=object)
    for chrom, sub_cpg in cpg_df.groupby("chrom"):
        order = np.argsort(sub_cpg["mid"].to_numpy())
        mids = sub_cpg["mid"].to_numpy()[order]
        ids = sub_cpg[id_col].to_numpy()[order]
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
        nearest_pos = np.where(use_right, mids[i_right], mids[i_left])
        nearest_id = np.where(use_right, ids[i_right], ids[i_left])
        idx = sub_peaks.index.to_numpy()
        out_chrom[idx] = chrom
        out_pos[idx] = nearest_pos
        out_id[idx] = nearest_id
    return out_chrom, out_pos, out_id


def build_hit_list(peaks_with_dist, importances, cpg_df, top_n, window,
                    cpg_id_col=None, gene_lookup=None, top_k=None, dedup_peaks=None):
    """
    peaks_with_dist: the `peaks` frame returned by run_enrichment (has dist_cpg, mid).
    gene_lookup: optional dict[peak_name-or-(chrom,start,end) -> gene symbol] OR a
                 callable(row) -> gene symbol, for annotating the PEAK side. If the
                 peaks frame has no natural name, pass gene_lookup=None and only
                 coordinates + nearest CpG id/distance are reported.
    top_k: if given, return only the top_k closest hits overall, sorted by dist_cpg
           -- use this for a "top K closest pairs" table.
    dedup_peaks: if True (default when top_k is set), the same genomic peak that is
           top-ranked for multiple cell types is collapsed into one row, with
           `label` becoming a comma-joined list of all cell types it appeared for
           -- otherwise a locus recurring in 6 cell types would eat 6 of the top_k
           slots with identical coordinates.
    Returns a tidy DataFrame: label, rank, chrom, start, end, [gene], cpg_chrom,
    cpg_pos, cpg_id, dist_cpg
    """
    if dedup_peaks is None:
        dedup_peaks = top_k is not None
    cpg_chrom, cpg_pos, cpg_id = nearest_cpg_info(peaks_with_dist, cpg_df, id_col=cpg_id_col)
    rows = []
    for label, imp in importances.items():
        order = np.argsort(imp)[::-1][:top_n]
        rank = {int(i): r for r, i in enumerate(order)}
        sub = peaks_with_dist.iloc[list(rank.keys())].copy()
        sub = sub[sub["dist_cpg"] <= window]
        if len(sub) == 0:
            continue
        sub["label"] = label
        sub["rank"] = [rank[i] for i in sub.index]
        sub["cpg_chrom"] = cpg_chrom[sub.index.to_numpy()]
        sub["cpg_pos"] = cpg_pos[sub.index.to_numpy()]
        sub["cpg_id"] = cpg_id[sub.index.to_numpy()]
        if gene_lookup is not None:
            sub["gene"] = [gene_lookup(r) for _, r in sub.iterrows()]
        rows.append(sub)
    cols = ["label", "rank", "chrom", "start", "end"]
    if gene_lookup is not None:
        cols.append("gene")
    cols += ["cpg_chrom", "cpg_pos", "cpg_id", "dist_cpg"]
    if not rows:
        return pd.DataFrame(columns=cols)
    hits = pd.concat(rows, ignore_index=True)
    hits = hits[cols].sort_values(["dist_cpg", "label"])
    if dedup_peaks:
        def _join_labels(s):
            uniq = sorted(set(s))
            if len(uniq) <= 3:
                return ", ".join(uniq)
            return ", ".join(uniq[:3]) + f" +{len(uniq) - 3} more"

        group_keys = ["chrom", "start", "end", "cpg_chrom", "cpg_pos", "cpg_id", "dist_cpg"]
        if gene_lookup is not None:
            group_keys.append("gene")
        merged = (hits.groupby(group_keys, as_index=False)
                       .agg(label=("label", _join_labels), rank=("rank", "min")))
        hits = merged[cols].sort_values(["dist_cpg", "chrom", "start"])
    if top_k is not None:
        hits = hits.head(top_k)
    return hits


def bh_qvalues(pvals):
    """Standard Benjamini-Hochberg FDR correction. Returns q-values, same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def add_bh_qvalues(df, group_cols, p_col="perm_p", q_col="q"):
    """BH-correct p_col within each group of group_cols (e.g. correct across cell
    types at each fixed top_n/window -- 'cross-CT pooled BH', matching the
    convention already used for the GREAT pathway-enrichment figures)."""
    df = df.copy()
    df[q_col] = np.nan
    for _, idx in df.groupby(group_cols).groups.items():
        df.loc[idx, q_col] = bh_qvalues(df.loc[idx, p_col].to_numpy())
    return df
