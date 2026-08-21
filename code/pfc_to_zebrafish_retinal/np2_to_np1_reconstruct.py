"""
Companion to np2_to_np1_convert.py. Run under numpy<2.0 (reprod env) to
reconstruct the original ndarray/Series/DataFrame/dict/list/tuple structure
from the plain-Python surrogate produced by the converter. Values are
bit-for-bit identical to the original numpy>=2.0 pickle; only the on-disk
serialization format changed.
"""
import pickle
import numpy as np
import pandas as pd


def reconstruct(obj):
    if isinstance(obj, dict) and "__kind__" in obj:
        kind = obj["__kind__"]
        if kind == "ndarray_raw":
            arr = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"]).copy()
            return arr
        if kind == "ndarray_object":
            arr = np.array(obj["data"], dtype=object).reshape(obj["shape"])
            return arr
        if kind == "Series":
            return pd.Series(reconstruct(obj["data"]), index=obj["index"], name=obj["name"])
        if kind == "DataFrame":
            data = {c: reconstruct(obj["data"][c]) for c in obj["columns"]}
            return pd.DataFrame(data, index=obj["index"], columns=obj["columns"])
        if kind == "dict":
            return {reconstruct(k): reconstruct(v) for k, v in obj["items"]}
        if kind == "tuple":
            return tuple(reconstruct(v) for v in obj["items"])
        raise ValueError(f"Unknown kind {kind}")
    if isinstance(obj, dict):
        return {reconstruct(k): reconstruct(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [reconstruct(v) for v in obj]
    return obj


def load_np2_pickle_compat(path):
    """Load a converted (plain-Python-surrogate) pickle and reconstruct it."""
    with open(path, "rb") as f:
        conv = pickle.load(f)
    return reconstruct(conv)


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    obj = load_np2_pickle_compat(p)
    print(type(obj), list(obj.keys()) if isinstance(obj, dict) else None)
