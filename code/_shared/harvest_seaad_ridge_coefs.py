"""
Standalone harvester: lm2factor_peak_contribution.py's per-label Ridge coefficient
training is cache-gated (`if label not in coef_cache`), so when
cache/ridge_coef_cache.pkl already exists from a prior run, log_coef() never
fires on rerun. This script extracts the same top-200-by-|coef| rows directly
from the cache pickle instead of forcing an expensive retrain.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("/homes/gws/pzy/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_coef

ROOT = Path(__file__).resolve().parent.parent / "seaad_alltissues_investigate"

with open(ROOT / "cache" / "ridge_coef_cache.pkl", "rb") as f:
    coef_cache = pickle.load(f)
with open(ROOT / "cache" / "shap_sex_pct_cache.pkl", "rb") as f:
    meta = pickle.load(f)
feat_names = meta["feat_names"]

for label, coef in coef_cache.items():
    coef = np.asarray(coef)
    for rank, i in enumerate(np.argsort(np.abs(coef))[::-1][:200]):
        log_coef(
            "seaad_alltissues_investigate", clock_name="dlpfc_xds_pfc357_peaks_ridge_clock",
            feature=feat_names[i], coefficient=float(coef[i]),
            modality="ATAC", cell_type=label, rank=rank, model_type="RidgeCV",
        )
    print(f"{label}: logged top 200 of {len(coef)} coefficients")
print("Done.")
