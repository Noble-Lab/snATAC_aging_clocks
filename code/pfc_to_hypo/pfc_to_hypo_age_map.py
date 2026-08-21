#!/usr/bin/env python3
"""
Build cache/intersect_all_map.pkl and cache/mouse_peak_liftover.pkl: the
mouse-peak -> PFC-peak feature map used by the PFC -> mouse hypothalamus ATAC
clock ("intersect_all" strategy).

Mouse hypothalamus peak midpoints are lifted mm10->hg38 (downloading the UCSC
chain file if not already cached) and intersected against the 521,217 PFC
peaks; only PFC peaks with >=1 overlapping lifted mouse peak are kept.

This cache is loaded directly by plot_violin_v2.py and compute_peak_importance.py.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr_lib
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
        logging.FileHandler(LOG_DIR / "run_age_map.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

PFC_ALL_CACHE = Path(
    "~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl"
)
CHAIN_FILE = CACHE / "mm10ToHg38.over.chain.gz"


def parse_peaks(names: list) -> pd.DataFrame:
    rows = []
    for p in names:
        p = str(p)
        if ":" in p:
            ch, rest = p.split(":", 1)
            a, b = rest.split("-")
        else:
            ch, a, b = p.rsplit("-", 2)
        rows.append((ch, int(a), int(b)))
    return pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])


def load_intersect_all_map() -> tuple:
    """
    Return (M_m, pfc_mask) for the intersect_all strategy.
    Uses cached liftover + overlap map (builds from scratch if absent).
    M_m  : csr (n_mouse_peaks, n_pfc_shared_feats)  — maps mouse peaks to PFC feat idx
    pfc_mask : bool array (n_pfc_peaks,) — which PFC peaks are shared
    """
    cache = CACHE / "intersect_all_map.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    import urllib.request
    from pyliftover import LiftOver

    CHAIN_URL = (
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver"
        "/mm10ToHg38.over.chain.gz"
    )
    if not CHAIN_FILE.exists():
        log.info("Downloading chain file…")
        urllib.request.urlretrieve(CHAIN_URL, CHAIN_FILE)

    lo = LiftOver(str(CHAIN_FILE))

    with open(PFC_ALL_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    pfc_peaks = list(pfc_all["peak_names"])

    with open(CACHE / "hypo_pseudobulks.pkl", "rb") as f:
        hypo = pickle.load(f)
    mouse_peaks = hypo["peak_names"]

    pfc_df   = parse_peaks(pfc_peaks)
    mouse_df = parse_peaks(mouse_peaks)
    n_mouse  = len(mouse_df)
    n_pfc    = len(pfc_peaks)

    # Liftover midpoints
    lo_cache = CACHE / "mouse_peak_liftover.pkl"
    if lo_cache.exists():
        with open(lo_cache, "rb") as f:
            lifted = pickle.load(f)
    else:
        log.info("Lifting mouse midpoints mm10→hg38…")
        lifted = []
        for ch, s, e in zip(mouse_df["Chromosome"], mouse_df["Start"], mouse_df["End"]):
            hit = lo.convert_coordinate(ch, (s + e) // 2)
            lifted.append((hit[0][0], int(hit[0][1])) if hit else None)
        with open(lo_cache, "wb") as f:
            pickle.dump(lifted, f, protocol=4)

    # Build PFC PyRanges
    pfc_df2 = pfc_df.copy()
    pfc_df2["pfc_idx"] = np.arange(n_pfc, dtype=np.int32)
    pfc_pr = pr_lib.PyRanges(pfc_df2)

    # Intersect
    valid = [(i, lifted[i]) for i in range(n_mouse) if lifted[i] is not None]
    q_df  = pd.DataFrame({
        "Chromosome": [p[0] for _, p in valid],
        "Start":      [p[1] for _, p in valid],
        "End":        [p[1] + 1 for _, p in valid],
        "mouse_idx":  [i for i, _ in valid],
    })
    ov      = pr_lib.PyRanges(q_df).join(pfc_pr)
    ov_df   = ov.df
    m_rows  = ov_df["mouse_idx"].values.astype(np.int32)
    p_idxs  = ov_df["pfc_idx"].values.astype(np.int32)

    uniq_pfc = np.unique(p_idxs)
    remap    = np.full(n_pfc, -1, dtype=np.int32)
    remap[uniq_pfc] = np.arange(len(uniq_pfc), dtype=np.int32)
    pfc_mask = np.zeros(n_pfc, dtype=bool)
    pfc_mask[uniq_pfc] = True

    M_m = sp.csr_matrix(
        (np.ones(len(m_rows), np.float32), (m_rows, remap[p_idxs])),
        shape=(n_mouse, len(uniq_pfc)),
    )
    log.info(f"  Intersect map: {len(uniq_pfc):,} shared PFC features")

    with open(cache, "wb") as f:
        pickle.dump((M_m, pfc_mask), f, protocol=4)
    return M_m, pfc_mask


def main():
    log.info("=" * 70)
    log.info("PFC -> mouse hypothalamus intersect-all feature map cache build")
    log.info("=" * 70)
    M_m, pfc_mask = load_intersect_all_map()
    log.info(f"Shared features: {int(pfc_mask.sum()):,}")
    log.info("Cache build complete.")


if __name__ == "__main__":
    main()
