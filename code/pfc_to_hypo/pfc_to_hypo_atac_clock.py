#!/usr/bin/env python3
"""
Build cache/hypo_pseudobulks.pkl: donor pseudobulks from the mouse hypothalamus
ATAC dataset (~/hypo_atac.h5ad, mm10 peaks, 20 samples, age groups Y=3mo/M=12mo/
A=24mo), aggregated both as all-cells-per-sample and per mouse cell type.

This cache is the shared input for the rest of the PFC -> mouse hypothalamus
pipeline (pfc_to_hypo_glu_gaba.py, pfc_to_hypo_age_map.py, plot_violin_v2.py,
compute_peak_importance.py), each of which loads it directly with pickle.
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
for _d in (CACHE, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

HYPO_H5AD  = Path("~/hypo_atac.h5ad")
CHUNK_SIZE = 5_000

MOUSE_CTS_ALL = [
    "Astrocytes", "Endothelial", "Ependymal", "Fibroblasts",
    "Hypendymal", "Immune", "Mural", "NG/OPC", "Neurons",
    "Oligodendrocytes", "ParsTuber", "Tanycytes",
]


def build_hypo_pseudobulks() -> dict:
    cache = CACHE / "hypo_pseudobulks.pkl"
    if cache.exists():
        log.info("Loading cached hypo pseudobulks …")
        with open(cache, "rb") as f:
            return pickle.load(f)

    log.info("Building hypo pseudobulks (scanning h5ad) …")
    adata = ad.read_h5ad(HYPO_H5AD, backed="r")
    obs   = adata.obs.copy()

    sample_meta = (
        obs[["Sample", "age"]]
        .drop_duplicates("Sample")
        .sort_values("Sample")
        .set_index("Sample")
    )
    samples     = sample_meta.index.tolist()
    age_letters = sample_meta["age"].tolist()
    N           = len(samples)
    s2i         = {s: i for i, s in enumerate(samples)}
    peak_names  = list(adata.var_names)
    n_peaks     = len(peak_names)

    avail_cts = sorted(
        ct for ct in MOUSE_CTS_ALL if ct in obs["major"].unique()
    )
    ct2i  = {ct: i for i, ct in enumerate(avail_cts)}
    n_cts = len(avail_cts)

    sum_all = np.zeros((N, n_peaks), dtype=np.float64)
    cc_all  = np.zeros(N, dtype=np.int64)
    sum_ct  = np.zeros((N, n_cts, n_peaks), dtype=np.float32)
    cc_ct   = np.zeros((N, n_cts), dtype=np.int64)

    cell_ids = adata.obs_names.values
    n_cells  = len(cell_ids)
    scodes   = obs["Sample"].map(s2i).values.astype(np.intp)
    ctcodes  = obs["major"].map(ct2i).fillna(-1).astype(int).values

    log.info(f"  {N} samples, {n_cells:,} cells, {n_peaks:,} peaks, {n_cts} CTs")
    t0 = time.time()

    for start in range(0, n_cells, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_cells)
        ids = cell_ids[start:end]
        X   = adata[ids].to_memory().X
        X   = (
            np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
        ).astype(np.float32)

        sc = scodes[start:end]
        ct = ctcodes[start:end]

        for si in np.unique(sc):
            mk = sc == si
            sum_all[si] += X[mk].sum(0).astype(np.float64)
            cc_all[si]  += int(mk.sum())

        for cti in range(n_cts):
            mc = ct == cti
            if not mc.any():
                continue
            for si in np.unique(sc[mc]):
                mk = mc & (sc == si)
                sum_ct[si, cti] += X[mk].sum(0)
                cc_ct[si, cti]  += int(mk.sum())

        if (start // CHUNK_SIZE) % 10 == 0:
            log.info(f"    {end:,}/{n_cells:,}  ({time.time()-t0:.1f}s)")

    adata.file.close()

    result = {
        "peak_names":  peak_names,
        "samples":     samples,
        "age_letters": age_letters,
        "all_cells":   {"sum": sum_all, "cc": cc_all},
        "per_ct": {
            ct: {
                "sum": sum_ct[:, i, :].astype(np.float64),
                "cc":  cc_ct[:, i],
            }
            for i, ct in enumerate(avail_cts)
        },
    }
    with open(cache, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info(f"  Cached → {cache}")
    return result


def main():
    log.info("=" * 70)
    log.info("Mouse hypothalamus ATAC pseudobulk cache build")
    log.info("=" * 70)
    build_hypo_pseudobulks()
    log.info("Cache build complete.")


if __name__ == "__main__":
    main()
