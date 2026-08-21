"""
Compat loader for sibling-project pickle caches that were built under a
numpy>=2.0 environment (their pickled ndarrays reference the internal
'numpy._core' module path introduced in numpy 2.0), while this project's
unified `reprod` env deliberately pins numpy==1.26.4 (numpy<2, where that
module is still named 'numpy.core'). Plain `pickle.load()` on such a file
raises `ModuleNotFoundError: No module named 'numpy._core.numeric'`.

load_pickle_compat(path) handles this in two tiers:

Tier 1: in a short-lived subprocess, alias numpy._core -> numpy.core, unpickle
the file, and re-pickle it as a plain numpy<2-native pickle. Running this in
a subprocess (rather than in-process) isolates a known failure mode: pickled
pandas DataFrames whose block manager holds numpy-2-specific dtype/structseq
metadata can SEGFAULT the interpreter during the shimmed unpickle.

Tier 2 (used only if tier 1's subprocess crashes): load the file natively
under a genuine numpy>=2.0 interpreter from another conda/micromamba env on
this machine (no shim, no ABI mismatch), recursively strip it down to plain
Python builtins (numpy arrays -> {"__ndarray__", dtype, tolist()}; pandas
DataFrames -> {"__dataframe__", columns, index, data}) with zero
numpy-version-specific pickle references, and reconstruct real
ndarrays/DataFrames from that representation back in the main process.

Converted copies are cached (by source mtime+size) under
pfc_internal_valid/results/_numpy2_converted_cache/; this module never
modifies anything inside the sibling project directories it reads from
(e.g. atac_processing_techniques/cache/).
"""
import pickle
import subprocess
import sys
from pathlib import Path

CONVERTED_DIR = Path("~/pfc_internal_valid/results/_numpy2_converted_cache")  # holds converted copies up to ~1GB, kept out of the repo

# Genuine numpy>=2.0 interpreters available on this machine (tier 2 fallback).
_NUMPY2_PYTHONS = [
    "~/micromamba/envs/NUMPY_2_INTERPRETER1_HERE/bin/python",
    "~/micromamba/envs/NUMPY_2_INTERPRETER2_HERE/bin/python",
]

_DUMP_SCRIPT_TIER1 = """
import sys, pickle
import numpy.core as core, numpy.core.numeric as numeric, numpy.core.multiarray as multiarray
sys.modules['numpy._core'] = core
sys.modules['numpy._core.numeric'] = numeric
sys.modules['numpy._core.multiarray'] = multiarray
with open({src!r}, 'rb') as f:
    obj = pickle.load(f)
with open({dst!r}, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
"""

# Tier-2 script: runs under a genuine numpy>=2 interpreter (no shim needed).
# Recursively strips the object down to plain Python builtins so the result
# has zero numpy-version-specific pickle references.
_DUMP_SCRIPT_TIER2 = r"""
import pickle
import numpy as np
import pandas as pd

def convert(obj):
    if isinstance(obj, pd.DataFrame):
        return {{"__dataframe__": True,
                 "columns": list(obj.columns),
                 "index": list(obj.index),
                 "data": {{c: convert(obj[c].to_numpy()) for c in obj.columns}}}}
    if isinstance(obj, np.ndarray):
        return {{"__ndarray__": True, "dtype": str(obj.dtype), "shape": list(obj.shape),
                 "data": obj.tolist()}}
    if isinstance(obj, dict):
        return {{k: convert(v) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return [convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

with open({src!r}, 'rb') as f:
    obj = pickle.load(f)
converted = convert(obj)
with open({dst!r}, 'wb') as f:
    pickle.dump(converted, f, protocol=4)
"""


def _rebuild(obj):
    """Inverse of the tier-2 convert(): plain-Python repr -> ndarray/DataFrame."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.asarray(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        if obj.get("__dataframe__"):
            data = {c: _rebuild(obj["data"][c]) for c in obj["columns"]}
            return pd.DataFrame(data, index=obj["index"])
        return {k: _rebuild(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebuild(v) for v in obj]
    return obj


def load_pickle_compat(path):
    """Load a pickle file, transparently handling numpy>=2.0-pickled arrays
    when running under numpy<2.0. Returns the unpickled object."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    converted = CONVERTED_DIR / f"{path.stem}.{stat.st_size}.{int(stat.st_mtime)}.pkl"
    rebuild_needed_marker = converted.with_suffix(converted.suffix + ".tier2")

    if converted.exists():
        print(f"  [numpy2_compat] using cached converted copy {converted}", flush=True)
        with open(converted, "rb") as f:
            obj = pickle.load(f)
        return _rebuild(obj) if rebuild_needed_marker.exists() else obj

    print(f"  [numpy2_compat] {path.name} needs numpy2->numpy1 conversion "
          f"(tier 1: isolated subprocess) -> {converted}", flush=True)
    script = _DUMP_SCRIPT_TIER1.format(src=str(path), dst=str(converted))
    r = subprocess.run([sys.executable, "-c", script])
    if r.returncode == 0:
        with open(converted, "rb") as f:
            return pickle.load(f)

    print(f"  [numpy2_compat] tier 1 crashed (rc={r.returncode}, likely a "
          f"pandas/numpy2 structseq ABI segfault) — falling back to tier 2 "
          f"(genuine numpy>=2 interpreter + plain-Python re-encoding)", flush=True)
    for py2 in _NUMPY2_PYTHONS:
        if not Path(py2).exists():
            continue
        script2 = _DUMP_SCRIPT_TIER2.format(src=str(path), dst=str(converted))
        r2 = subprocess.run([py2, "-c", script2])
        if r2.returncode == 0:
            rebuild_needed_marker.touch()
            with open(converted, "rb") as f:
                obj = pickle.load(f)
            return _rebuild(obj)
        print(f"  [numpy2_compat] tier 2 via {py2} failed (rc={r2.returncode})", flush=True)

    raise RuntimeError(
        f"Could not load {path} under numpy<2.0: tier 1 (shim+subprocess) "
        f"segfaulted and tier 2 (genuine numpy2 interpreter) also failed or "
        f"was unavailable. Tried: {_NUMPY2_PYTHONS}"
    )
