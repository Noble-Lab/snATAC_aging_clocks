"""Cross-dataset ATAC peak clock: train on PFC-357 healthy donors (peak features),
test on SEA-AD DLPFC 43-donor cohort.

Feature sources
  All-cells : pfc_peak_pseudobulk.pkl (357 donors, 521K 500bp tiles)
               × seaad_peak_pseudobulk.pkl (218K called peaks)
               → coordinate-based interval overlap → ~197K common features
               PFC training value = sum of overlapping 500bp tiles per called peak
  Per-CT    : pfc_peak_pseudobulk_by_ct.pkl  (357 × 197576, pre-aligned)
               × seaad_peak_pseudobulk_by_ct.pkl (same feature space)
               → use directly (already the same 197576 peaks)

Donors are z-scored per-donor before training / prediction.

Outputs (results/):
  xds_pfc357_peaks_{all_cells|CT}_all.csv   — predictions + metadata (donor_id,
                                               true_age, pred_age, residual, braak)
  xds_pfc357_peaks_summary.csv              — summary r/MAE per cell type

Downstream, plot_adnc_residuals.py and plot_adnc_violin_sexsplit_twopanel.py
reload the per-cell-type CSVs and plot residual vs ADNC severity.
"""
from __future__ import annotations

import bisect
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
CACHE_DIR  = Path("~/atac_processing_techniques/cache")
GA_CACHE   = Path("~/age_accel_per_cell_type/cache")
RES        = ROOT / "results"
FIG        = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PFC_AC_PKL  = CACHE_DIR / "pfc_peak_pseudobulk.pkl"
PFC_CT_PKL  = CACHE_DIR / "pfc_peak_pseudobulk_by_ct.pkl"
SEA_AC_PKL  = CACHE_DIR / "seaad_peak_pseudobulk.pkl"
SEA_CT_PKL  = CACHE_DIR / "seaad_peak_pseudobulk_by_ct.pkl"
SEAAD_META_PKL = GA_CACHE / "seaad_pseudobulks_by_ct.pkl"

CT_MAP = {
    "Excitatory": "Exc",
    "Inhibitory": "Inh",
    "Oligo":      "Oligo",
    "Astro":      "Astro",
    "Microglia":  "Mic",
    "OPC":        "OPC",
}

BRAAK_ORDER = ["Braak 0", "Braak I", "Braak II", "Braak III",
               "Braak IV", "Braak V", "Braak VI"]
BRAAK_NUM   = {b: i for i, b in enumerate(BRAAK_ORDER)}

# ── helpers ────────────────────────────────────────────────────────────────────

def zscore_per_donor(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def train_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    feat_idx = None
    if top_k is not None and X_train.shape[1] > top_k:
        cors = np.array([
            stats.pearsonr(X_train[:, j], y_train)[0]
            for j in range(X_train.shape[1])
        ])
        feat_idx = np.argsort(np.abs(cors))[::-1][:top_k]
        X_train  = X_train[:, feat_idx]
        X_test   = X_test[:,  feat_idx]
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    model = RidgeCV(alphas=alphas, cv=min(5, len(X_train)))
    model.fit(X_train, y_train)
    return model.predict(X_test), feat_idx


def ols_residuals(true_age, pred_age):
    slope, intercept, *_ = stats.linregress(true_age, pred_age)
    return pred_age - (slope * true_age + intercept), float(slope), float(intercept)


def compute_overlap_mapping(
    pfc_peak_names: list[str],
    sea_peak_names: list[str],
) -> tuple[np.ndarray, list[list[int]]]:
    """Find all SEA-AD called peaks that overlap ≥1 PFC 500bp tile.

    Returns
    -------
    valid_sea_idx : 1-D int array — indices into sea_peak_names
    sea_to_pfc    : list of same length, each entry is a list of PFC tile indices
    """
    # group PFC tiles by chromosome, sorted by start
    pfc_by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for i, p in enumerate(pfc_peak_names):
        try:
            ch, rest = p.split(":", 1)
            s, e = rest.split("-")
            pfc_by_chrom[ch].append((int(s), int(e), i))
        except Exception:
            continue
    pfc_starts_by_chrom: dict[str, list[int]] = {}
    for ch in pfc_by_chrom:
        pfc_by_chrom[ch].sort()
        pfc_starts_by_chrom[ch] = [s for s, e, i in pfc_by_chrom[ch]]

    valid_sea_idx: list[int] = []
    sea_to_pfc: list[list[int]] = []

    for j, p in enumerate(sea_peak_names):
        try:
            ch, rest = p.split(":", 1)
            ps, pe = rest.split("-")
            ps, pe = int(ps), int(pe)
        except Exception:
            continue
        tiles = pfc_by_chrom.get(ch)
        if not tiles:
            continue
        starts = pfc_starts_by_chrom[ch]
        # tiles with start < pe
        right = bisect.bisect_left(starts, pe)
        overlapping: list[int] = []
        for k in range(right - 1, -1, -1):
            ts, te, ti = tiles[k]
            if te > ps:
                overlapping.append(ti)
            elif ts < ps - 600:  # 500bp tile + small buffer
                break
        if overlapping:
            valid_sea_idx.append(j)
            sea_to_pfc.append(overlapping)

    return np.array(valid_sea_idx, dtype=np.int64), sea_to_pfc


def build_pfc_overlap_matrix(
    pfc_matrix: np.ndarray,
    sea_to_pfc: list[list[int]],
    n_donors: int,
) -> np.ndarray:
    """For each SEA-AD peak, sum overlapping PFC tile signals → (donors, n_common)."""
    n_feat = len(sea_to_pfc)
    out = np.zeros((n_donors, n_feat), dtype=np.float32)
    for col, tile_idxs in enumerate(sea_to_pfc):
        out[:, col] = pfc_matrix[:, tile_idxs].sum(axis=1)
    return out


# ── run_combo (unchanged from GA script) ──────────────────────────────────────

def run_combo(
    label: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test: np.ndarray,
    donors_test: list[str],
    braak_series: pd.Series,
    top_k: int | None,
    regime_tag: str,
    tag_prefix: str = "peaks",
) -> pd.DataFrame:
    tag = f"{label}_{regime_tag}"
    print(f"  {tag}: n_train={len(X_train)} n_test={len(X_test)} features={X_train.shape[1]}", end="")

    preds, _ = train_predict(X_train, y_train, X_test, top_k=top_k)

    r   = float(stats.pearsonr(y_test, preds)[0])
    mae = float(np.abs(y_test - preds).mean())
    print(f"  →  r={r:+.3f}  MAE={mae:.1f}y")

    residuals, slope, intercept = ols_residuals(y_test, preds)

    df = pd.DataFrame({
        "donor_id": donors_test,
        "true_age": y_test,
        "pred_age": preds,
        "residual": residuals,
    })
    df["braak"]     = df["donor_id"].map(braak_series)
    df["braak_num"] = df["braak"].map(BRAAK_NUM)

    df.to_csv(RES / f"xds_pfc357_{tag_prefix}_{tag}.csv", index=False)

    return pd.DataFrame([{
        "label": label, "regime": regime_tag,
        "n_train": len(X_train), "n_test": len(X_test),
        "n_features_used": (top_k if top_k and X_train.shape[1] > top_k else X_train.shape[1]),
        "r": r, "mae": mae,
        "ols_slope": slope, "ols_intercept": intercept,
    }])


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading PFC-357 peak pseudobulks ...")
    with open(PFC_AC_PKL, "rb") as f:
        pfc_ac = pickle.load(f)
    with open(PFC_CT_PKL, "rb") as f:
        pfc_ct = pickle.load(f)

    pfc_donors = np.array(pfc_ac["donors"])
    pfc_ages   = pfc_ac["age"].astype(float)
    pfc_matrix = pfc_ac["peak_sum"].astype(np.float32)   # (357, 521217)
    pfc_peaks  = pfc_ac["peak_names"]

    valid_tr = ~np.isnan(pfc_ages)
    pfc_donors_ok = pfc_donors[valid_tr]
    y_train   = pfc_ages[valid_tr]
    pfc_matrix_ok = pfc_matrix[valid_tr]
    print(f"  PFC donors (non-NaN age): {len(pfc_donors_ok)}  tiles: {pfc_matrix_ok.shape[1]}")

    print("\nLoading SEA-AD peak pseudobulks ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    with open(SEA_CT_PKL, "rb") as f:
        sea_ct = pickle.load(f)

    sea_donors = np.array(sea_ac["donors"])
    sea_ages   = sea_ac["age"].astype(float)
    sea_matrix = sea_ac["peak_sum"].astype(np.float32)  
    sea_peaks  = sea_ac["peak_names"]
    print(f"  SEA-AD donors: {len(sea_donors)}  peaks: {sea_matrix.shape[1]}")

    # Filter to DLPFC 43 donors
    man = pd.read_csv(ROOT / "cache" / "donor_manifest_annot.csv")
    dlpfc43 = set(man[
        (man.tissue == "DLPFC") & (man.modality == "snatac") &
        (man.kind == "fragments") & man.donor_id.notna()
    ]["donor_id"])
    dlpfc_mask = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_donors = sea_donors[dlpfc_mask]
    dlpfc_ages   = sea_ages[dlpfc_mask]
    dlpfc_matrix = sea_matrix[dlpfc_mask]
    print(f"  DLPFC donors: {dlpfc_mask.sum()}  age {dlpfc_ages.min():.0f}–{dlpfc_ages.max():.0f}")

    # Braak staging from GA meta
    # The donor-metadata cache was pickled under numpy>=2.0; _seaad_meta_compat.py loads
    # a numpy-version-agnostic copy of it instead of unpickling it directly.
    from _seaad_meta_compat import load_seaad_meta
    braak_series = load_seaad_meta()["Braak"]   # indexed by donor_id
    print(f"  Braak dist: {braak_series.loc[dlpfc_donors].value_counts().to_dict()}")

    valid_te = ~np.isnan(dlpfc_ages)
    test_donors = dlpfc_donors[valid_te]
    y_test      = dlpfc_ages[valid_te]
    Xte_all     = dlpfc_matrix[valid_te]

    # ── All-cells: interval overlap ──────────────────────────────────────────
    overlap_cache = ROOT / "cache" / "pfc521k_x_seaad218k_overlap.npz"
    if overlap_cache.exists():
        print("\nLoading cached interval overlap ...")
        c = np.load(overlap_cache, allow_pickle=True)
        valid_sea_idx = c["valid_sea_idx"]
        sea_to_pfc    = list(c["sea_to_pfc"])
    else:
        print("\nComputing interval overlap (PFC 521K tiles × SEA-AD 218K peaks) ...")
        valid_sea_idx, sea_to_pfc = compute_overlap_mapping(pfc_peaks, sea_peaks)
        np.savez(overlap_cache, valid_sea_idx=valid_sea_idx,
                 sea_to_pfc=np.array(sea_to_pfc, dtype=object))
        print(f"  Saved overlap cache.")
    print(f"  SEA-AD peaks with ≥1 overlapping PFC tile: {len(valid_sea_idx)}")

    print("\nBuilding PFC overlap matrix ...")
    Xtr_all = build_pfc_overlap_matrix(pfc_matrix_ok, sea_to_pfc, len(pfc_donors_ok))
    Xte_all = Xte_all[:, valid_sea_idx]
    print(f"  Xtr shape: {Xtr_all.shape}  Xte shape: {Xte_all.shape}")

    print("\nNormalising PFC-357 (all-cells) ...")
    Xtr_z_ac = zscore_per_donor(Xtr_all.astype(np.float64))
    Xte_z_ac = zscore_per_donor(Xte_all.astype(np.float64))

    # ── Run all-cells combos ─────────────────────────────────────────────────
    print("\n─── All-cells ───")
    all_rows = [run_combo(
        "all_cells", Xtr_z_ac, y_train, Xte_z_ac, y_test,
        list(test_donors), braak_series, top_k=None, regime_tag="all",
    )]

    # ── Per-CT combos ────────────────────────────────────────────────────────
    print("\n─── Per cell type (197576 pre-aligned peaks) ───")
    for ct_key, ct_short in CT_MAP.items():
        if ct_key not in pfc_ct or ct_key not in sea_ct:
            print(f"  [{ct_short}] skipped — key missing")
            continue

        pfc_ct_mat = pfc_ct[ct_key].astype(np.float32) 
        sea_ct_mat = sea_ct[ct_key].astype(np.float32) 
        # donors come from pfc_ac / sea_ac donor lists in same row order
        tr_ok = valid_tr.copy()
        te_dlpfc = dlpfc_mask.copy()

        # drop donors where the pseudobulk row is all-NaN or all-zero (absent CT)
        pfc_ct_ok  = pfc_ct_mat[valid_tr]   # same row selection as pfc_donors_ok
        sea_ct_dlpfc = sea_ct_mat[dlpfc_mask][valid_te]

        # mask donors where >50% of features are NaN or row is all-zero
        tr_nonzero = (pfc_ct_ok > 0).any(axis=1)
        te_nonzero = (sea_ct_dlpfc > 0).any(axis=1)

        Xtr_ct = pfc_ct_ok[tr_nonzero].astype(np.float64)
        Xte_ct = sea_ct_dlpfc[te_nonzero].astype(np.float64)
        y_tr_ct = y_train[tr_nonzero]
        y_te_ct = y_test[te_nonzero]
        d_te_ct = list(test_donors[te_nonzero])

        if len(Xtr_ct) < 10 or len(Xte_ct) < 3:
            print(f"  [{ct_short}] skipped — too few donors (train={len(Xtr_ct)} test={len(Xte_ct)})")
            continue

        Xtr_z = zscore_per_donor(Xtr_ct)
        Xte_z = zscore_per_donor(Xte_ct)

        print(f"\n  [{ct_short}] n_train={len(Xtr_ct)} n_test={len(Xte_ct)}")
        all_rows.append(run_combo(
            ct_short, Xtr_z, y_tr_ct, Xte_z, y_te_ct,
            d_te_ct, braak_series, top_k=None, regime_tag="all",
        ))

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(RES / "xds_pfc357_peaks_summary.csv", index=False)
    print(f"\nwrote xds_pfc357_peaks_summary.csv")
    print(summary[["label", "regime", "n_test", "r", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
