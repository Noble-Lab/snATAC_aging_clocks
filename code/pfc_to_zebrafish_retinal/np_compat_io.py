"""
Compatibility loader for pickles serialized under numpy>=2.0 (some upstream
sibling-project caches this project reuses as-is, e.g.
age_accel_per_cell_type/cache/v3/pfc_pseudobulks_by_ct_v3.pkl and
atac_processing_techniques/cache/pfc_peak_pseudobulk_by_ct_native.pkl), while
this project's `reprod` env pins numpy==1.26.4. numpy>=2.0 pickles ndarrays
via a reduce path referencing the internal `numpy._core` module, which does
not exist in numpy<2.0 (there it's `numpy.core`), so a direct load raises
`ModuleNotFoundError: No module named 'numpy._core.numeric'`; aliasing
numpy._core -> numpy.core in sys.modules segfaults partway through unpickling
rather than raising a catchable error, so it is not a viable workaround.

safe_pickle_load(path) does a plain pickle.load() and, only on that specific
ModuleNotFoundError, falls back to a pre-converted numpy-agnostic copy at
cache/np1compat/<name>.np1compat.pkl -- a plain-Python surrogate (nested
dict/list/bytes, no numpy-specific reduce logic) produced once under a
numpy>=2.0 interpreter by np2_to_np1_convert.py, and reconstructed back into
real ndarrays/DataFrames by np2_to_np1_reconstruct.load_np2_pickle_compat().
For every other file (pickled under numpy<2.0) this is a no-op pass-through.
"""
import pickle
from pathlib import Path

_COMPAT_DIR = Path("~/pfc_to_zebrafish_retinal/cache/np1compat")


def safe_pickle_load(path):
    path = Path(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise
        compat = _COMPAT_DIR / (path.stem + ".np1compat.pkl")
        if not compat.exists():
            raise RuntimeError(
                f"{path} was pickled under numpy>=2.0 and cannot be loaded "
                f"under the active numpy<2.0 interpreter, and no pre-converted "
                f"compat copy exists at {compat}. Regenerate it by running "
                f"np2_to_np1_convert.py under an env with numpy>=2.0 "
            ) from e
        from np2_to_np1_reconstruct import load_np2_pickle_compat
        return load_np2_pickle_compat(str(compat))
