"""
Build the caches needed for the peak-level PFC -> hippocampus ATAC clock:

  1. cache/hip_peak_pseudobulks_in_pfc_space.pkl   hippocampus donor pseudobulks,
     projected into PFC peak space (each hippocampus donor has its own set of
     60k-163k called peaks; a per-donor overlap matrix against the 521,217 PFC
     peaks projects donor counts onto the shared PFC peak axis)
  2. cache/pfc_per_ct_peak_pseudobulks.pkl         PFC donor pseudobulks, per
     broad cell type, restricted to the PFC peaks with >=1 overlapping
     hippocampus donor peak ("intersecting" peaks)

PFC: 357 healthy donors, 521,217 called peaks (final_atac_data.h5ad).
Hippocampus: GSE278576, 40 donors, per-donor called peaks (raw 10x h5 files).

The Ridge clock that consumes these caches is trained downstream, in
compute_peak_importance.py.

External large-file dependency (kept at a fixed absolute path, not inside this
repo): both caches this script writes (99MB and 579MB) live under
~/pfc_to_hippocampus/cache. Everything else (figures/, logs/) is
local to this script's own directory.
"""
import gc
import h5py
import logging
import pickle
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT  = Path(__file__).resolve().parent
CACHE = Path("~/pfc_to_hippocampus/cache")  # large caches (100s of MB), kept out of the repo
LOG   = ROOT / "logs"
for d in (CACHE, LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG / "pfc_to_hip_atac_clock_peaks.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

# ── paths ─────────────────────────────────────────────────────────────────────
_PFC_CACHE_TMP = Path("/tmp/pfc_peak_pseudobulk.pkl")
_PFC_CACHE_NFS = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PFC_CACHE = _PFC_CACHE_TMP if _PFC_CACHE_TMP.exists() else _PFC_CACHE_NFS
PFC_H5AD  = Path("~/data_back/PFC_brain_multiome/final_atac_data.h5ad")
PFC_RNA   = Path("~/data_back/PFC_brain_multiome/final_rna_data.h5ad")
HIP_RAW   = Path("~/data_back/GSE278576_hippocampus/raw")
HIP_META  = Path("~/data_back/GSE278576_hippocampus/metadata.tsv.gz")

CHUNK_SIZE = 10_000   # for PFC h5ad scan

# ── cell type maps ────────────────────────────────────────────────────────────
PFC_CT_MAP = {
    "ExN": "Excitatory", "InN": "Inhibitory",
    "Oligo": "Oligo",    "Astro": "Astro",
    "MG": "Microglia",   "OPC": "OPC",
}
HIP_CT_MAP = {
    "DG": "Excitatory",  "CA1": "Excitatory",  "CA2-CA3": "Excitatory",
    "SUB": "Excitatory", "NR2F2": "Excitatory",
    "SST": "Inhibitory", "VIP": "Inhibitory",   "PVALB": "Inhibitory",
    "LAMP5": "Inhibitory", "Chandelier": "Inhibitory",
    "Oligo": "Oligo",    "Astro": "Astro",
    "Microglia": "Microglia", "Macro": "Microglia",
    "OPC": "OPC",
}
BROAD_CTS = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


# ── peak parser + overlap matrix ─────────────────────────────────────────────
def parse_peaks(peak_names):
    chroms, starts, ends = [], [], []
    for p in peak_names:
        p = str(p)
        if ":" in p:
            chrom, rest = p.split(":", 1)
            a, b = rest.split("-", 1)
        else:
            bits = p.rsplit("-", 2); chrom = bits[0]; a = bits[1]; b = bits[2]
        chroms.append(chrom); starts.append(int(a)); ends.append(int(b))
    return np.array(chroms), np.array(starts, np.int32), np.array(ends, np.int32)


def compute_overlap_matrix(peaks_a, peaks_b):
    """
    Sparse (n_peaks_a × n_peaks_b) binary overlap matrix.
    Fully vectorized per chromosome — no Python loop over individual peaks.
    """
    ca, sa, ea = parse_peaks(peaks_a)
    cb, sb, eb = parse_peaks(peaks_b)
    chroms = np.unique(np.concatenate([ca, cb]))
    all_rows, all_cols = [], []

    for chrom in chroms:
        ai = np.where(ca == chrom)[0]
        bi = np.where(cb == chrom)[0]
        if len(ai) == 0 or len(bi) == 0:
            continue

        sa_c, ea_c = sa[ai], ea[ai]
        sb_c, eb_c = sb[bi], eb[bi]

        # Sort B by start for searchsorted
        ob    = np.argsort(sb_c)
        sb_s  = sb_c[ob]; eb_s = eb_c[ob]; bi_s = bi[ob]

        # Sort A by start
        oa    = np.argsort(sa_c)
        sa_s  = sa_c[oa]; ea_s = ea_c[oa]; ai_s = ai[oa]

        n_a, n_b = len(sa_s), len(sb_s)

        # For each A peak: upper = first B index where sb >= ea (condition 1 upper bound)
        upper = np.searchsorted(sb_s, ea_s, side="left")    # (n_a,)

        # Lower bound: B peaks ending before sa[i] can't overlap A[i].
        # Since B sorted by sb (not eb), use max B length as conservative bound:
        # any B peak with sb >= sa[i] - max_b_len might still have eb > sa[i].
        max_b_len = int((eb_s - sb_s).max()) if n_b > 0 else 5000
        lower = np.maximum(
            np.searchsorted(sb_s, sa_s - max_b_len, side="left"), 0
        )                                                    # (n_a,)

        # Number of candidate B peaks per A peak
        counts = np.maximum(upper - lower, 0).astype(np.int64)
        total  = int(counts.sum())
        if total == 0:
            continue

        # Build flat candidate index arrays (fully vectorized, no Python loop)
        cumsum = np.concatenate([[0], np.cumsum(counts)])       # (n_a+1,)
        a_rep  = np.repeat(np.arange(n_a, dtype=np.int64), counts)  # (total,)
        b_idx  = lower[a_rep] + (
            np.arange(total, dtype=np.int64) - cumsum[a_rep]
        )                                                        # (total,)

        # Exact filter: B must end after A starts (second overlap condition)
        valid  = eb_s[b_idx] > sa_s[a_rep]
        if valid.any():
            all_rows.extend(ai_s[a_rep[valid]].tolist())
            all_cols.extend(bi_s[b_idx[valid]].tolist())

    return sp.csr_matrix(
        (np.ones(len(all_rows), np.float32), (all_rows, all_cols)),
        shape=(len(peaks_a), len(peaks_b)),
    )


# ── normalization ─────────────────────────────────────────────────────────────
def build_hip_pseudobulks_in_pfc_space(pfc_peaks):
    """
    For each hippocampus donor:
      - load h5, filter to QC barcodes from metadata
      - aggregate cells → (n_hip_peaks_d,) pseudobulk (all-cells and per broad CT)
      - compute overlap matrix: (n_hip_peaks_d, n_pfc_peaks)
      - project to PFC peak space and accumulate

    Returns dict with keys:
      donors, age,
      all_cells: {peak_sum (N, n_pfc), cell_counts (N,)}
      per_ct: {CT: {peak_sum, cell_counts}, ...}
    """
    cache_path = CACHE / "hip_peak_pseudobulks_in_pfc_space.pkl"
    if cache_path.exists():
        log.info("Loading cached hippocampus pseudobulks (PFC peak space)")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log.info("Building hippocampus pseudobulks in PFC peak space…")
    n_pfc = len(pfc_peaks)

    # Load metadata
    meta_raw  = pd.read_csv(HIP_META, sep="\t")
    meta_deep = meta_raw[meta_raw["bacrode"].str.contains("_deep_")].copy()
    meta_deep["bacrode"] = meta_deep["bacrode"].str.replace("_deep_", "_", regex=False)
    meta = pd.concat(
        [meta_raw[~meta_raw["bacrode"].str.contains("_deep_")], meta_deep],
        ignore_index=True,
    ).set_index("bacrode")
    meta["broad_ct"] = meta["subclass"].map(HIP_CT_MAP)

    donors_age = (meta[["orig.ident", "Age"]]
                  .drop_duplicates("orig.ident")
                  .rename(columns={"orig.ident": "donor_id", "Age": "age"})
                  .set_index("donor_id")["age"])
    donors = sorted(donors_age.index.tolist())
    d2idx  = {d: i for i, d in enumerate(donors)}
    N      = len(donors)
    age    = donors_age.reindex(donors).values.astype(float)
    n_cts  = len(BROAD_CTS)

    # Pre-index metadata by donor for fast per-donor barcode lookup (avoids
    # np.isin on 615k object strings which takes 2+ min per donor)
    meta_by_donor = {}
    for d, grp in meta.groupby("orig.ident"):
        prefix = d + "_"
        raw_bcs = grp.index.str[len(prefix):]
        meta_by_donor[d] = (set(raw_bcs), grp)

    # Accumulators in PFC peak space
    peak_sum_all = np.zeros((N, n_pfc), dtype=np.float32)
    cc_all       = np.zeros(N, dtype=np.int64)
    peak_sum_ct  = np.zeros((N, n_cts, n_pfc), dtype=np.float32)
    cc_ct        = np.zeros((N, n_cts), dtype=np.int64)

    # Non-empty h5 files
    h5_files = {}
    for fp in sorted(HIP_RAW.glob("GSE278576_hc*_raw_feature_bc_matrix.h5")):
        if fp.stat().st_size == 0:
            continue
        donor = fp.stem.split("_")[1]
        h5_files[donor] = fp

    t0 = time.time()
    for donor in sorted(h5_files.keys()):
        if donor not in d2idx:
            log.warning(f"  {donor}: not in metadata, skipping")
            continue
        d_idx = d2idx[donor]
        t_d   = time.time()

        # Load ATAC peaks from h5
        log.info(f"  {donor}: loading h5…")
        hip_peak_names, raw_barcodes, X_peaks = load_h5_atac_peaks(h5_files[donor])
        log.info(f"  {donor}: h5 loaded ({X_peaks.shape}, {time.time()-t_d:.1f}s)")
        # X_peaks: (n_raw_barcodes, n_hip_peaks_d)

        # Filter to QC-passed barcodes via per-donor small set (fast O(1) lookup)
        log.info(f"  {donor}: filtering {len(raw_barcodes)} barcodes…")
        donor_raw_set, donor_meta = meta_by_donor.get(donor, (set(), None))
        if donor_meta is None:
            log.warning(f"  {donor}: no metadata group, skipping")
            continue
        valid_mask = np.array([bc in donor_raw_set for bc in raw_barcodes], dtype=bool)
        log.info(f"  {donor}: barcode filter done ({valid_mask.sum()} valid, {time.time()-t_d:.1f}s)")
        n_valid    = valid_mask.sum()
        if n_valid == 0:
            log.warning(f"  {donor}: no matching barcodes in metadata")
            continue

        X_valid    = X_peaks[valid_mask]       # (n_valid, n_hip_peaks_d)
        valid_full = np.array([f"{donor}_{bc}" for bc in raw_barcodes[valid_mask]])
        broad_cts  = meta.loc[valid_full, "broad_ct"].values

        # Pseudobulk in donor peak space
        all_sum = np.asarray(X_valid.sum(axis=0), dtype=np.float64).ravel()  # (n_hip_peaks_d,)
        ct_sums = {}
        for cti, ct in enumerate(BROAD_CTS):
            ct_mask = broad_cts == ct
            if ct_mask.any():
                ct_sums[ct] = np.asarray(X_valid[ct_mask].sum(axis=0),
                                         dtype=np.float64).ravel()
                cc_ct[d_idx, cti] += int(ct_mask.sum())
        cc_all[d_idx] += n_valid

        del X_peaks, X_valid
        gc.collect()

        # Per-donor overlap matrix: (n_hip_peaks_d, n_pfc_peaks)
        log.info(f"  {donor}: computing overlap matrix ({len(hip_peak_names)} hip × {len(pfc_peaks)} pfc)…")
        M_d = compute_overlap_matrix(list(hip_peak_names.astype(str)), list(pfc_peaks))
        log.info(f"  {donor}: overlap done ({M_d.nnz} nnz, {time.time()-t_d:.1f}s)")
        # M_d.T: (n_pfc_peaks, n_hip_peaks_d) → M_d: (n_hip_peaks_d, n_pfc_peaks)

        # Project to PFC space: sum_d (n_hip_peaks_d,) @ M_d → (n_pfc_peaks,)
        pfc_proj = np.asarray(all_sum @ M_d, dtype=np.float32).ravel()
        peak_sum_all[d_idx] += pfc_proj

        for cti, ct in enumerate(BROAD_CTS):
            if ct in ct_sums:
                pfc_ct_proj = np.asarray(ct_sums[ct] @ M_d, dtype=np.float32).ravel()
                peak_sum_ct[d_idx, cti] += pfc_ct_proj

        del M_d, pfc_proj, all_sum, ct_sums
        gc.collect()

        log.info(f"  {donor}: {n_valid} cells, {len(hip_peak_names)} peaks  "
                 f"({time.time()-t_d:.0f}s, total {time.time()-t0:.0f}s)")

    log.info(f"  Total: {cc_all.sum():,} cells in {time.time()-t0:.0f}s")

    result = {
        "donors":    donors,
        "age":       age,
        "all_cells": {"peak_sum": peak_sum_all, "cell_counts": cc_all},
        "per_ct": {
            ct: {"peak_sum": peak_sum_ct[:, i, :], "cell_counts": cc_ct[:, i]}
            for i, ct in enumerate(BROAD_CTS)
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache_path}")
    return result


# ── PFC per-CT pseudobulk in PFC peak space (subset to keep_idx) ─────────────
def build_pfc_per_ct_peak_pseudobulks(keep_idx):
    """
    Scan PFC h5ad, aggregate by (donor, broad_ct), return only keep_idx peaks.
    Cached as pfc_per_ct_peak_pseudobulks.pkl.
    """
    cache_path = CACHE / "pfc_per_ct_peak_pseudobulks.pkl"
    if cache_path.exists():
        log.info("Loading cached PFC per-CT pseudobulks (PFC peak space)")
        with open(cache_path, "rb") as f:
            res = pickle.load(f)
        # Verify keep_idx matches
        if res.get("n_keep") == len(keep_idx):
            return res
        log.info("  keep_idx size changed, rebuilding")

    log.info("Building PFC per-CT pseudobulks from h5ad (scanning 14GB)…")
    n_keep = len(keep_idx)

    rna = ad.read_h5ad(PFC_RNA, backed="r")
    sid_age = rna.obs.groupby("SampleID", observed=True)["Age"].first().to_dict()
    rna.file.close()

    a   = ad.read_h5ad(PFC_H5AD, backed="r")
    obs = a.obs.copy()
    obs["broad_ct"] = obs["cell_type"].map(PFC_CT_MAP)
    obs["age"]      = obs["sample_id"].map(sid_age)
    obs_valid       = obs[obs["age"].notna() & obs["broad_ct"].notna()].copy()

    donors = sorted(obs_valid["sample_id"].unique())
    d2idx  = {d: i for i, d in enumerate(donors)}
    ct2idx = {ct: i for i, ct in enumerate(BROAD_CTS)}
    N      = len(donors)
    n_cts  = len(BROAD_CTS)
    age    = np.array([sid_age[d] for d in donors], dtype=float)

    peak_sums = np.zeros((N, n_cts, n_keep), dtype=np.float32)
    cc        = np.zeros((N, n_cts), dtype=np.int64)

    cell_ids = obs_valid.index.values
    dcodes   = obs_valid["sample_id"].map(d2idx).values.astype(np.intp)
    ctcodes  = obs_valid["broad_ct"].map(ct2idx).values.astype(np.intp)
    n_cells  = len(cell_ids)

    log.info(f"  PFC: {N} donors, {n_cells:,} valid cells, n_keep={n_keep:,}")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        sub = a[ids].to_memory()
        X   = sub.X
        if sp.issparse(X):
            X = X.tocsr()
        else:
            X = sp.csr_matrix(X)
        # Slice to keep_idx peaks
        Xk = np.asarray(X[:, keep_idx].todense(), dtype=np.float32)

        chunk_d  = dcodes[start:end]
        chunk_ct = ctcodes[start:end]

        for cti in range(n_cts):
            mask_ct = chunk_ct == cti
            if not mask_ct.any():
                continue
            for d in np.unique(chunk_d[mask_ct]):
                mask = mask_ct & (chunk_d == d)
                peak_sums[d, cti] += Xk[mask].sum(axis=0)
                cc[d, cti]        += mask.sum()

        if (start // CHUNK_SIZE) % 20 == 0:
            log.info(f"    pfc {end:,}/{n_cells:,}  ({time.time()-t0:.0f}s)")

    a.file.close()
    log.info(f"  Done in {time.time()-t0:.0f}s")

    result = {
        "n_keep": n_keep,
        "donors": donors,
        "age":    age,
        "per_ct": {
            ct: {"peak_sum": peak_sums[:, i, :], "cell_counts": cc[:, i]}
            for i, ct in enumerate(BROAD_CTS)
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache_path}")
    return result




def main():
    with open(PFC_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    pfc_peaks = pfc_all["peak_names"]
    log.info(f"PFC: {len(pfc_all['age'])} donors x {len(pfc_peaks):,} peaks")

    hip = build_hip_pseudobulks_in_pfc_space(pfc_peaks)
    col_sum  = hip["all_cells"]["peak_sum"].sum(axis=0)
    keep_idx = np.where(col_sum > 0)[0]
    log.info(f"Intersecting PFC peaks: {len(keep_idx):,} / {len(pfc_peaks):,}")

    build_pfc_per_ct_peak_pseudobulks(keep_idx)
    log.info("Cache build complete.")


if __name__ == "__main__":
    main()
