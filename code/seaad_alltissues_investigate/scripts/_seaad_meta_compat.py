"""
Loads the SEA-AD donor metadata table from a numpy-version-agnostic cache.

The shared cache `age_accel_per_cell_type/cache/seaad_pseudobulks_by_ct.pkl`
was pickled under numpy>=2.0, while this project's `reprod` env pins
numpy==1.26.4; numpy<2 cannot unpickle numpy-2.x ndarrays directly (raises
ModuleNotFoundError, and a sys.modules alias workaround segfaults rather than
raising a catchable exception, since numpy 2.x's ndarray __reduce__ payload
isn't binary-compatible with numpy 1.26's _reconstruct). Since all downstream
scripts only need the small `meta` DataFrame out of
that pickle -- never the multi-GB rna/atac blocks -- cache/seaad_meta_compat.pkl
stores just `meta`, pre-extracted to pure-Python str/float/None values with no
numpy objects at all, so it loads cleanly under any numpy version.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

COMPAT_PKL = Path(__file__).resolve().parent.parent / "cache" / "seaad_meta_compat.pkl"


def load_seaad_meta() -> pd.DataFrame:
    """Return the SEA-AD donor metadata DataFrame (84 donors x 14 cols),
    indexed by donor id, without touching the numpy-2-pickled original."""
    with open(COMPAT_PKL, "rb") as f:
        payload = pickle.load(f)
    meta = pd.DataFrame.from_dict(payload["data"], orient="index")
    meta = meta[payload["columns"]]
    meta.index.name = payload["index_name"]
    return meta
