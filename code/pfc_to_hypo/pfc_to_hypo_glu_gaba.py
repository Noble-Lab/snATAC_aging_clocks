#!/usr/bin/env python3
"""
Build cache/hypo_glu_gaba_pseudobulks.pkl: mouse hypothalamus donor
pseudobulks for the two neuronal subtypes (GLU, GABA), split out of the
h5ad's "Neurons" population via its major2 column. GLU is matched to the PFC
Excitatory model and GABA to the PFC Inhibitory model downstream.

This cache is merged into cache/hypo_pseudobulks.pkl's per-CT dict by
plot_violin_v2.py and compute_peak_importance.py, which load it directly.
"""

import logging
import pickle
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT    = Path(__file__).resolve().parent
CACHE   = ROOT / "cache"
LOG_DIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_glu_gaba.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

HYPO_H5AD  = Path("~/hypo_atac.h5ad")
CHUNK_SIZE = 5_000


def build_glu_gaba_pseudobulks(samples, age_letters, n_peaks):
    """
    Scan hypo h5ad using major2 column to build GLU and GABA pseudobulks.
    Uses sparse matrix ops to avoid dense conversion.
    Returns: {"GLU": {"sum": (N,n_peaks), "cc": (N,)},
              "GABA": {"sum": (N,n_peaks), "cc": (N,)}}
    """
    cache = CACHE / "hypo_glu_gaba_pseudobulks.pkl"
    if cache.exists():
        log.info("Loading cached GLU/GABA pseudobulks …")
        with open(cache, "rb") as f:
            return pickle.load(f)

    log.info("Building GLU/GABA pseudobulks (sparse, scanning h5ad) …")
    adata = ad.read_h5ad(HYPO_H5AD, backed="r")
    obs   = adata.obs.copy()

    N   = len(samples)
    s2i = {s: i for i, s in enumerate(samples)}

    sum_glu  = np.zeros((N, n_peaks), dtype=np.float64)
    sum_gaba = np.zeros((N, n_peaks), dtype=np.float64)
    cc_glu   = np.zeros(N, dtype=np.int64)
    cc_gaba  = np.zeros(N, dtype=np.int64)

    cell_ids = adata.obs_names.values
    n_cells  = len(cell_ids)
    scodes   = obs["Sample"].map(s2i).values.astype(np.intp)
    m2_vals  = obs["major2"].values   # 'GLU', 'GABA', or other

    log.info(f"  {n_cells:,} cells, {n_peaks:,} peaks")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        X   = adata[ids].to_memory().X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.astype(np.float32)

        sc  = scodes[start:end]
        m2  = m2_vals[start:end]

        for label, s_arr, cc_arr in [("GLU", sum_glu, cc_glu),
                                      ("GABA", sum_gaba, cc_gaba)]:
            mask_ct = m2 == label
            if not mask_ct.any():
                continue
            X_ct = X[mask_ct]
            sc_ct = sc[mask_ct]
            for si in np.unique(sc_ct):
                mask_si = sc_ct == si
                s_arr[si]  += np.asarray(X_ct[mask_si].sum(0)).ravel().astype(np.float64)
                cc_arr[si] += int(mask_si.sum())

        if (start // CHUNK_SIZE) % 10 == 0:
            log.info(f"    {end:,}/{n_cells:,}  ({time.time()-t0:.1f}s)")

    adata.file.close()
    result = {
        "GLU":  {"sum": sum_glu,  "cc": cc_glu},
        "GABA": {"sum": sum_gaba, "cc": cc_gaba},
    }
    with open(cache, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache}  ({time.time()-t0:.0f}s total)")
    return result


def main():
    log.info("=" * 70)
    log.info("Mouse hypothalamus GLU/GABA pseudobulk cache build")
    log.info("=" * 70)

    with open(CACHE / "hypo_pseudobulks.pkl", "rb") as f:
        hypo = pickle.load(f)
    samples     = hypo["samples"]
    age_letters = hypo["age_letters"]
    n_peaks     = len(hypo["peak_names"])
    log.info(f"  {len(samples)} samples, {n_peaks:,} peaks")

    build_glu_gaba_pseudobulks(samples, age_letters, n_peaks)
    log.info("Cache build complete.")


if __name__ == "__main__":
    main()
