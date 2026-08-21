"""
Build the caches needed to train a PFC -> hippocampus ATAC age clock:

  1. cache/pfc_peaks_to_hip_tiles_M.pkl        PFC-peak x hippocampus-tile overlap matrix
  2. cache/hip_pseudobulks.pkl                 hippocampus donor pseudobulks (all-cells + per broad cell type)
  3. cache/pfc_per_ct_tile_pseudobulks.pkl     PFC donor pseudobulks in tile space, per broad cell type

PFC: 357 healthy donors (10x Multiome ATAC, called peaks).
Hippocampus: GSE278576, 40 donors (5kb ATAC tiles).

PFC peak counts are projected into hippocampus tile space via the peak-tile overlap
matrix (1), so both datasets end up on the same feature axis (the tiles hit by >=1 PFC
peak). The Ridge clocks that consume these caches are trained downstream, in
plot_scatter_tile_zscore_horiz_nr2f2inh.py and pfc_to_hip_atac_clock_peaks.py.

Broad cell type mapping:
  PFC        : ExN->Excitatory, InN->Inhibitory, Oligo, Astro, MG->Microglia, OPC
  Hippocampus: DG/CA1/CA2-CA3/SUB/NR2F2->Excitatory,
               SST/VIP/PVALB/LAMP5/Chandelier->Inhibitory,
               Oligo, Astro, Microglia/Macro, OPC

External large-file dependency (kept at a fixed absolute path, not inside this
repo, since these caches run 100MB-600MB): cache/ under
~/pfc_to_hippocampus/cache -- see cache/pfc_per_ct_peak_pseudobulks.pkl
(579MB) and cache/hip_peak_pseudobulks_in_pfc_space.pkl (99MB), built by
pfc_to_hip_atac_clock_peaks.py. Everything else (figures/, logs/) is local to
this script's own directory.
"""
import logging
import pickle
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
CACHE = Path("~/pfc_to_hippocampus/cache")  # large caches (100s of MB), kept out of the repo
LOG = ROOT / "logs"
for d in (CACHE, LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG / "pfc_to_hip_atac_clock.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

# ── paths ─────────────────────────────────────────────────────────────────────
PFC_CACHE = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PFC_H5AD  = Path("~/data_back/PFC_brain_multiome/final_atac_data.h5ad")
PFC_RNA   = Path("~/data_back/PFC_brain_multiome/final_rna_data.h5ad")
HIP_ATAC  = Path("~/data_back/GSE278576_hippocampus/processed/atac_tiles.h5ad")
HIP_META  = Path("~/data_back/GSE278576_hippocampus/metadata.tsv.gz")

CHUNK_SIZE = 10_000

# ── cell type maps ────────────────────────────────────────────────────────────
PFC_CT_MAP = {
    "ExN": "Excitatory", "InN": "Inhibitory",
    "Oligo": "Oligo",   "Astro": "Astro",
    "MG": "Microglia",  "OPC": "OPC",
}
HIP_CT_MAP = {
    "DG": "Excitatory",  "CA1": "Excitatory",  "CA2-CA3": "Excitatory",
    "SUB": "Excitatory", "NR2F2": "Excitatory",
    "SST": "Inhibitory", "VIP": "Inhibitory",   "PVALB": "Inhibitory",
    "LAMP5": "Inhibitory", "Chandelier": "Inhibitory",
    "Oligo": "Oligo",   "Astro": "Astro",
    "Microglia": "Microglia", "Macro": "Microglia",
    "OPC": "OPC",
}
BROAD_CTS = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


# ── peak/tile parser ──────────────────────────────────────────────────────────
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


def compute_overlap_matrix(peaks_a, tiles_b):
    """Sparse (n_peaks_a x n_tiles_b) binary overlap matrix."""
    ca, sa, ea = parse_peaks(peaks_a)
    cb, sb, eb = parse_peaks(tiles_b)
    chroms = np.unique(np.concatenate([ca, cb]))
    rows, cols = [], []
    for chrom in chroms:
        ai = np.where(ca == chrom)[0]
        bi = np.where(cb == chrom)[0]
        if len(ai) == 0 or len(bi) == 0:
            continue
        ob = np.argsort(sb[bi])
        bi_s = bi[ob]; sb_s = sb[bi][ob]; eb_s = eb[bi][ob]
        for i in ai:
            s, e = int(sa[i]), int(ea[i])
            r = int(np.searchsorted(sb_s, e, side="left"))
            if r == 0:
                continue
            msk = eb_s[:r] > s
            if not msk.any():
                continue
            for j in bi_s[:r][msk]:
                rows.append(int(i)); cols.append(int(j))
    return sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(len(peaks_a), len(tiles_b)),
    )


# ── hippocampus pseudobulk builder ────────────────────────────────────────────
def build_hip_pseudobulks():
    cache_path = CACHE / "hip_pseudobulks.pkl"
    if cache_path.exists():
        log.info("Loading cached hippocampus pseudobulks")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log.info("Building hippocampus pseudobulks from atac_tiles.h5ad…")
    adata = ad.read_h5ad(HIP_ATAC, backed="r")
    tile_names = list(adata.var_names.astype(str))
    n_tiles = len(tile_names)

    # Build barcode → metadata mapping (handle _deep_ donors)
    meta_raw = pd.read_csv(HIP_META, sep="\t")
    meta_deep = meta_raw[meta_raw["bacrode"].str.contains("_deep_")].copy()
    meta_deep["bacrode"] = meta_deep["bacrode"].str.replace("_deep_", "_", regex=False)
    meta = pd.concat([meta_raw[~meta_raw["bacrode"].str.contains("_deep_")], meta_deep],
                     ignore_index=True).set_index("bacrode")

    # Build obs augmented with broad_ct
    obs = adata.obs.copy()
    obs["subclass"] = meta.reindex(obs.index)["subclass"].values
    obs["broad_ct"] = obs["subclass"].map(HIP_CT_MAP)

    donors_age = (obs[["donor_id", "age"]].drop_duplicates("donor_id")
                  .set_index("donor_id")["age"])
    donors = sorted(donors_age.index.tolist())
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    N = len(donors)
    age = donors_age.reindex(donors).values.astype(float)

    n_cts = len(BROAD_CTS)

    tile_sum_all = np.zeros((N, n_tiles), dtype=np.float64)
    cc_all       = np.zeros(N, dtype=np.int64)
    tile_sum_ct  = np.zeros((N, n_cts, n_tiles), dtype=np.float64)
    cc_ct        = np.zeros((N, n_cts), dtype=np.int64)

    # Precompute per-cell donor/CT codes (full array)
    dcodes_full  = obs["donor_id"].map(donor_to_idx).values.astype(np.intp)
    ct_vals_full = obs["broad_ct"].values                 # str or NaN

    cell_ids = adata.obs_names.values
    n_cells  = len(cell_ids)
    log.info(f"  Hippocampus: {N} donors, {n_cells:,} cells, {n_tiles} tiles")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        X = adata[ids].to_memory().X
        if sp.issparse(X):
            X = np.asarray(X.todense(), dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        dcodes = dcodes_full[start:end]
        ct_vals = ct_vals_full[start:end]

        # Vectorised scatter-add for all-cells
        for d in np.unique(dcodes):
            mask_d = dcodes == d
            tile_sum_all[d] += X[mask_d].sum(axis=0)
            cc_all[d]       += mask_d.sum()

        # Vectorised scatter-add per CT
        for cti, ct in enumerate(BROAD_CTS):
            mask_ct = (ct_vals == ct)
            if not mask_ct.any():
                continue
            for d in np.unique(dcodes[mask_ct]):
                mask_dc = mask_ct & (dcodes == d)
                tile_sum_ct[d, cti] += X[mask_dc].sum(axis=0)
                cc_ct[d, cti]       += mask_dc.sum()

        if (start // CHUNK_SIZE) % 10 == 0:
            log.info(f"    hip {end:,}/{n_cells:,}  ({time.time()-t0:.0f}s)")

    adata.file.close()
    log.info(f"  Done in {time.time()-t0:.0f}s")

    result = {
        "tile_names": tile_names,
        "donors": donors,
        "age": age,
        "all_cells": {"tile_sum": tile_sum_all, "cell_counts": cc_all},
        "per_ct": {
            ct: {"tile_sum": tile_sum_ct[:, i, :], "cell_counts": cc_ct[:, i]}
            for i, ct in enumerate(BROAD_CTS)
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache_path}")
    return result


# ── PFC per-CT pseudobulk builder (projects peaks → tiles on-the-fly) ────────
def build_pfc_per_ct_tile_pseudobulks(M_keep, n_keep_tiles):
    """
    Scan PFC h5ad once, project each cell's peak vector to tile space via M_keep,
    aggregate by (donor, broad_ct). Cached after first run.
    M_keep: sparse (n_pfc_peaks × n_keep_tiles)
    """
    cache_path = CACHE / "pfc_per_ct_tile_pseudobulks.pkl"
    if cache_path.exists():
        log.info("Loading cached PFC per-CT tile pseudobulks")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log.info("Building PFC per-CT pseudobulks in tile space (scanning 14GB h5ad)…")

    # Load age from RNA obs
    rna = ad.read_h5ad(PFC_RNA, backed="r")
    sid_age = rna.obs.groupby("SampleID", observed=True)["Age"].first().to_dict()
    rna.file.close()

    a = ad.read_h5ad(PFC_H5AD, backed="r")
    obs = a.obs.copy()
    obs["broad_ct"] = obs["cell_type"].map(PFC_CT_MAP)
    obs["age"]      = obs["sample_id"].map(sid_age)
    obs_valid = obs[obs["age"].notna() & obs["broad_ct"].notna()].copy()

    donors     = sorted(obs_valid["sample_id"].unique())
    d2idx      = {d: i for i, d in enumerate(donors)}
    ct2idx     = {ct: i for i, ct in enumerate(BROAD_CTS)}
    N          = len(donors)
    n_cts      = len(BROAD_CTS)
    age        = np.array([sid_age[d] for d in donors], dtype=float)

    tile_sums  = np.zeros((N, n_cts, n_keep_tiles), dtype=np.float64)
    cc         = np.zeros((N, n_cts), dtype=np.int64)

    # Precompute codes for all valid cells (order = obs_valid.index)
    cell_ids   = obs_valid.index.values
    dcodes     = obs_valid["sample_id"].map(d2idx).values.astype(np.intp)
    ctcodes    = obs_valid["broad_ct"].map(ct2idx).values.astype(np.intp)
    n_cells    = len(cell_ids)

    M_csc = M_keep.tocsc()
    log.info(f"  PFC: {N} donors, {n_cells:,} valid cells, M_keep: {M_keep.shape}")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        sub = a[ids].to_memory()
        X = sub.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.astype(np.float32)

        # Project: (n_cells, n_peaks) @ (n_peaks, n_tiles) → (n_cells, n_tiles)
        X_tiles = np.asarray((X @ M_csc).todense(), dtype=np.float32)

        chunk_d  = dcodes[start:end]
        chunk_ct = ctcodes[start:end]

        # Vectorised scatter per (donor × ct)
        for cti in range(n_cts):
            mask_ct = chunk_ct == cti
            if not mask_ct.any():
                continue
            for d in np.unique(chunk_d[mask_ct]):
                mask = mask_ct & (chunk_d == d)
                tile_sums[d, cti] += X_tiles[mask].sum(axis=0)
                cc[d, cti]        += mask.sum()

        if (start // CHUNK_SIZE) % 20 == 0:
            log.info(f"    pfc {end:,}/{n_cells:,}  ({time.time()-t0:.0f}s)")

    a.file.close()
    log.info(f"  Done in {time.time()-t0:.0f}s")

    result = {
        "donors": donors,
        "age":    age,
        "per_ct": {
            ct: {"tile_sum": tile_sums[:, i, :], "cell_counts": cc[:, i]}
            for i, ct in enumerate(BROAD_CTS)
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache_path}")
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("PFC → Hippocampus ATAC cache build")
    log.info("=" * 70)

    # ── Load PFC all-cells pseudobulk ────────────────────────────────────────
    log.info("Loading PFC all-cells pseudobulk…")
    with open(PFC_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    pfc_peaks = pfc_all["peak_names"]
    pfc_sum   = pfc_all["peak_sum"].astype(np.float64)   # (357, 521217)
    log.info(f"  PFC: {len(pfc_all['age'])} donors × {len(pfc_peaks):,} peaks")

    # ── Hippocampus tile names ────────────────────────────────────────────────
    log.info("Loading hippocampus tile names…")
    _tmp = ad.read_h5ad(HIP_ATAC, backed="r")
    hip_tile_names = list(_tmp.var_names.astype(str))
    _tmp.file.close()
    log.info(f"  {len(hip_tile_names)} 5kb tiles")

    # ── Overlap matrix: PFC peaks → hippocampus tiles ────────────────────────
    ov_cache = CACHE / "pfc_peaks_to_hip_tiles_M.pkl"
    if ov_cache.exists():
        log.info("Loading cached overlap matrix…")
        with open(ov_cache, "rb") as f:
            M = pickle.load(f)
    else:
        log.info("Computing PFC peaks → hippocampus tile overlap matrix…")
        t0 = time.time()
        M = compute_overlap_matrix(pfc_peaks, hip_tile_names)
        log.info(f"  M shape: {M.shape}, nnz: {M.nnz:,}  ({time.time()-t0:.0f}s)")
        with open(ov_cache, "wb") as f:
            pickle.dump(M, f, protocol=4)

    # Keep only tiles with ≥1 overlapping PFC peak
    col_sum   = np.asarray(M.sum(axis=0)).ravel()
    keep_idx  = np.where(col_sum > 0)[0]
    M_keep    = M[:, keep_idx]   # (n_pfc_peaks, n_intersecting_tiles)
    n_keep    = len(keep_idx)
    log.info(f"  Intersecting tiles: {n_keep} / {len(hip_tile_names)}")

    # ── Project PFC all-cells pseudobulk to tile space (sanity check only; not cached) ──
    pfc_tile_sum = np.asarray(pfc_sum @ M_keep, dtype=np.float64)  # (357, n_keep)
    log.info(f"  PFC tile matrix: {pfc_tile_sum.shape}")

    # ── Hippocampus pseudobulks (cached) ─────────────────────────────────────
    build_hip_pseudobulks()

    # ── PFC per-CT pseudobulks in tile space (cached; scans 14GB h5ad) ───────
    build_pfc_per_ct_tile_pseudobulks(M_keep, n_keep)

    log.info("Cache build complete.")


if __name__ == "__main__":
    main()
