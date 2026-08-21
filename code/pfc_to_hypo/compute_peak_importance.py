#!/usr/bin/env python3
"""
Ridge-based peak importance ranking for the PFC -> Mouse Hypothalamus ATAC
clock (all_pfc strategy: all 357 PFC donors, full intersect-all feature
space, k=84,537 shared hg38 peaks).

The hypothalamus clock is a RidgeCV model. To rank
peaks analogously to PFC's own mean-|SHAP| pipeline, we fit one RidgeCV
model per condition (All_cells + 6 broad cell types) on PFC training data
restricted to the peaks shared with mouse hypothalamus, and rank peaks by
|ridge.coef_ * feature_std| (standardized-effect importance).

Loads:
  ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
  ~/atac_processing_techniques/cache/pfc_peak_pseudobulk_by_ct_native.pkl
  cache/intersect_all_map.pkl   (pfc_mask: 84,537/521,217 shared peaks)

Saves:
  results/peak_importance.pkl
    {condition: {"peak_names": np.array(str)[k], "importance": np.array(float)[k]}}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

# numpy<2 compat shim: cache/pfc_peak_pseudobulk_by_ct_native.pkl (shared
# atac_processing_techniques cache) was pickled under numpy>=2.0. numpy
# 1.26.4 (pinned by this repo's reproducibility env) ships a small
# numpy/_core/ forward-compat stub package for exactly this situation, but
# that stub covers multiarray/_multiarray_umath/umath/_dtype/_internal and
# NOT numpy._core.numeric -- and scipy/sklearn eagerly import numpy._core
# at their own import time, so by the time this script's pickle.load() runs,
# `numpy._core` already exists (as the incomplete stub) and a naive
# `if not hasattr(np, "_core")` guard would never fire. Patch in the missing
# `numeric` submodule explicitly. No-op under real numpy>=2.0.
try:
    import numpy._core.numeric  # noqa: F401
except ImportError:
    import numpy._core as _np_core_pkg
    import numpy.core.numeric as _np_core_numeric
    sys.modules["numpy._core.numeric"] = _np_core_numeric
    _np_core_pkg.numeric = _np_core_numeric

ROOT    = Path(__file__).resolve().parent
CACHE   = ROOT / "cache"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PFC_ALL_CACHE = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PFC_CT_CACHE  = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk_by_ct_native.pkl")

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_coef

RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))
CTS = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


def cpm_log1p(X):
    d = X.sum(axis=1, keepdims=True)
    d = np.where(d < 1.0, 1.0, d)
    return np.log1p(X / d * 1e6)


def zscore_per_donor(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def main():
    print("Loading PFC all-cells pseudobulk…", flush=True)
    with open(PFC_ALL_CACHE, "rb") as f:
        pfc_all = pickle.load(f)
    peak_names = np.array(pfc_all["peak_names"])
    age        = pfc_all["age"].astype(float)
    all_sum    = np.asarray(pfc_all["peak_sum"], dtype=np.float64)

    print("Loading intersect-all mask (mouse hypothalamus <-> PFC)…", flush=True)
    with open(CACHE / "intersect_all_map.pkl", "rb") as f:
        _M_m, pfc_mask = pickle.load(f)
    print(f"  {pfc_mask.sum():,} / {len(pfc_mask):,} shared PFC peaks", flush=True)

    print("Loading PFC per-CT pseudobulk (native)…", flush=True)
    with open(PFC_CT_CACHE, "rb") as f:
        pfc_ct = pickle.load(f)

    peak_names_kept = peak_names[pfc_mask]
    results = {}

    print("\n--- All_cells ---", flush=True)
    X = zscore_per_donor(cpm_log1p(all_sum[:, pfc_mask]))
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X, age)
    imp = np.abs(ridge.coef_ * X.std(axis=0))
    results["All_cells"] = {"peak_names": peak_names_kept, "importance": imp}
    print(f"  All_cells: n={X.shape[0]} k={X.shape[1]:,} alpha={ridge.alpha_:.3g} "
          f"R2={ridge.score(X, age):.3f}", flush=True)
    for rank, i in enumerate(np.argsort(imp)[::-1][:200]):
        log_coef("pfc_to_hypo", clock_name="pfc_to_hypo_all_pfc_ridge_clock",
                 feature=peak_names_kept[i], coefficient=float(ridge.coef_[i]),
                 modality="ATAC", cell_type="All_cells", importance=float(imp[i]),
                 rank=rank, model_type="RidgeCV")

    for ct in CTS:
        raw = np.asarray(pfc_ct[ct], dtype=np.float64)[:, pfc_mask]
        valid = raw.sum(axis=1) > 0
        X_ct = zscore_per_donor(cpm_log1p(raw[valid]))
        y_ct = age[valid]
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_ct, y_ct)
        imp = np.abs(ridge.coef_ * X_ct.std(axis=0))
        results[ct] = {"peak_names": peak_names_kept, "importance": imp}
        print(f"  {ct}: n={valid.sum()} k={X_ct.shape[1]:,} alpha={ridge.alpha_:.3g} "
              f"R2={ridge.score(X_ct, y_ct):.3f}", flush=True)
        for rank, i in enumerate(np.argsort(imp)[::-1][:200]):
            log_coef("pfc_to_hypo", clock_name="pfc_to_hypo_all_pfc_ridge_clock",
                     feature=peak_names_kept[i], coefficient=float(ridge.coef_[i]),
                     modality="ATAC", cell_type=ct, importance=float(imp[i]),
                     rank=rank, model_type="RidgeCV")

    out_path = RESULTS / "peak_importance.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
