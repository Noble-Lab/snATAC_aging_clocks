#!/usr/bin/env python3
"""
Ridge-based peak importance ranking for the PFC -> Human Hippocampus ATAC
clock (all_pfc strategy: all 357 PFC donors, full intersecting-peak feature
space, k=286,927 PFC peaks with >=1 hippocampus donor overlap).

The hippocampus clock is a RidgeCV model. To rank
peaks analogously to PFC's own mean-|SHAP| pipeline, we fit one RidgeCV
model per condition (All_cells + 6 broad cell types) on PFC training data
restricted to the peaks shared with hippocampus, and rank peaks by
|ridge.coef_ * feature_std| (standardized-effect importance).

External large-file dependencies (kept at fixed absolute paths, not inside
this repo):
  ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl (238MB)
  ~/pfc_to_hippocampus/cache/hip_peak_pseudobulks_in_pfc_space.pkl (99MB, to derive the intersecting mask)
  ~/pfc_to_hippocampus/cache/pfc_per_ct_peak_pseudobulks.pkl (579MB, PFC per-CT, already sliced to keep_idx)

Saves:
  results/peak_importance.pkl
    {condition: {"peak_names": np.array(str)[k], "importance": np.array(float)[k]}}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_coef

ROOT    = Path(__file__).resolve().parent
CACHE   = Path("~/pfc_to_hippocampus/cache")  # large caches (100s of MB), kept out of the repo
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PFC_ALL_CACHE = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")

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

    print("Deriving intersecting-peak mask (hippocampus <-> PFC)…", flush=True)
    with open(CACHE / "hip_peak_pseudobulks_in_pfc_space.pkl", "rb") as f:
        hip = pickle.load(f)
    col_sum  = hip["all_cells"]["peak_sum"].sum(axis=0)
    keep_idx = np.where(col_sum > 0)[0]
    mask = np.zeros(len(peak_names), dtype=bool)
    mask[keep_idx] = True
    print(f"  {mask.sum():,} / {len(mask):,} shared PFC peaks", flush=True)

    print("Loading PFC per-CT pseudobulk (already sliced to keep_idx)…", flush=True)
    with open(CACHE / "pfc_per_ct_peak_pseudobulks.pkl", "rb") as f:
        pfc_ct = pickle.load(f)
    assert pfc_ct["n_keep"] == len(keep_idx), (
        f"keep_idx size mismatch: cache={pfc_ct['n_keep']} vs recomputed={len(keep_idx)}"
    )

    peak_names_kept = peak_names[mask]
    results = {}

    print("\n--- All_cells ---", flush=True)
    X = zscore_per_donor(cpm_log1p(all_sum[:, mask]))
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X, age)
    imp = np.abs(ridge.coef_ * X.std(axis=0))
    results["All_cells"] = {"peak_names": peak_names_kept, "importance": imp}
    print(f"  All_cells: n={X.shape[0]} k={X.shape[1]:,} alpha={ridge.alpha_:.3g} "
          f"R2={ridge.score(X, age):.3f}", flush=True)
    for rank, i in enumerate(np.argsort(imp)[::-1][:200]):
        log_coef("pfc_to_hippocampus", clock_name="pfc_to_hip_all_pfc_ridge_clock",
                 feature=peak_names_kept[i], coefficient=float(ridge.coef_[i]),
                 modality="ATAC", cell_type="All_cells", importance=float(imp[i]),
                 rank=rank, model_type="RidgeCV")

    for ct in CTS:
        raw = np.asarray(pfc_ct["per_ct"][ct]["peak_sum"], dtype=np.float64)
        cc  = pfc_ct["per_ct"][ct]["cell_counts"]
        valid = cc > 0
        X_ct = zscore_per_donor(cpm_log1p(raw[valid]))
        y_ct = age[valid]
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_ct, y_ct)
        imp = np.abs(ridge.coef_ * X_ct.std(axis=0))
        results[ct] = {"peak_names": peak_names_kept, "importance": imp}
        print(f"  {ct}: n={valid.sum()} k={X_ct.shape[1]:,} alpha={ridge.alpha_:.3g} "
              f"R2={ridge.score(X_ct, y_ct):.3f}", flush=True)
        for rank, i in enumerate(np.argsort(imp)[::-1][:200]):
            log_coef("pfc_to_hippocampus", clock_name="pfc_to_hip_all_pfc_ridge_clock",
                     feature=peak_names_kept[i], coefficient=float(ridge.coef_[i]),
                     modality="ATAC", cell_type=ct, importance=float(imp[i]),
                     rank=rank, model_type="RidgeCV")

    out_path = RESULTS / "peak_importance.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
