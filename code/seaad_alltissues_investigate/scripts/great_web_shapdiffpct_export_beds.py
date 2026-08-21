"""Export top-N peaks by Δ percentile-rank mean |SHAP| (top-D vs bottom-D
OLS-residual donors), hg38, as BED files for submission to the REAL GREAT
web server (great.stanford.edu) via rGREAT's submitGreatJob() in
scripts/great_web_dlpfc_sweep.R.

Peaks are ranked by Δ percentile-rank mean |SHAP| (top-D vs bottom-D
OLS-residual donors), swept over D in N_DONORS_LIST and N in
TOP_N_PEAKS_LIST, and exported as raw genomic regions instead of being
collapsed to nearest-TSS gene symbols first -- GREAT assigns its own
regulatory-domain-based gene mapping.

The RidgeCV fit / prediction / OLS residual (which is the expensive step)
depends only on the label, not on D -- only the top-D/bottom-D donor split
does. So each label is trained exactly once here and reused across every D
in N_DONORS_LIST.

Loads: cache/pfc_peak_pseudobulk.pkl, cache/pfc_peak_pseudobulk_by_ct.pkl,
       cache/seaad_peak_pseudobulk.pkl, cache/seaad_peak_pseudobulk_by_ct.pkl,
       cache/pfc521k_x_seaad218k_overlap.npz, cache/donor_manifest_annot.csv
Saves: results/great_web_dlpfc_beds/{label}__shapdiffpct_top{D}_top{N}.bed
       (skips a BED that already exists, so re-running only fills in new D/N)
"""
from __future__ import annotations

import gc
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from shap_dlpfc_analysis import (
    ROOT,
    PFC_AC_PKL, PFC_CT_PKL, SEA_AC_PKL, SEA_CT_PKL,
    OVERLAP_NPZ,
    ALPHAS, ALL_LABELS, CT_MAP,
    zscore_per_donor,
    build_pfc_overlap_matrix,
)
from shap_diff_residual_donors import ols_residuals
from shap_pct_diff_residual_donors import shap_percentile_diff

N_DONORS_LIST    = [5]     # top / bottom OLS-residual donor counts
TOP_N_PEAKS_LIST = [2000]  # top-2000-SHAP-diff-ranked peaks exported per label

BED_DIR = ROOT / "results" / "great_web_dlpfc_beds"
BED_DIR.mkdir(parents=True, exist_ok=True)


def parse_peak(name: str):
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def write_bed(label: str, n_donors: int, top_n_peaks: int,
              feat_names: list[str], top_idx: np.ndarray) -> None:
    bed_path = BED_DIR / f"{label}__shapdiffpct_top{n_donors}_top{top_n_peaks}.bed"
    if bed_path.exists():
        print(f"  [{label}] top{n_donors}_top{top_n_peaks}: already exists, skipping")
        return
    with open(bed_path, "w") as f:
        for i in top_idx:
            chrom, start, end = parse_peak(feat_names[i])
            f.write(f"{chrom}\t{start}\t{end}\n")
    print(f"  [{label}] {len(top_idx)} peaks -> {bed_path}")


def process_label(
    label: str,
    Xtr_z: np.ndarray,
    y_tr: np.ndarray,
    Xte_z: np.ndarray,
    ages_te: np.ndarray,
    feat_names: list[str],
) -> None:
    n_te     = len(ages_te)
    max_don  = max(N_DONORS_LIST)
    if n_te < 2 * max_don:
        print(f"  [{label}] note — only {n_te} test donors (need >= {2*max_don} for "
              f"the largest D={max_don}; smaller D values will still be attempted)")

    m     = RidgeCV(alphas=ALPHAS).fit(Xtr_z, y_tr)
    preds = m.predict(Xte_z)
    resid = ols_residuals(ages_te.astype(float), preds)
    order = np.argsort(resid)
    baseline = Xtr_z.mean(0)

    for n_donors in N_DONORS_LIST:
        if n_te < 2 * n_donors:
            print(f"  [{label}] D={n_donors} skipped — only {n_te} test donors "
                  f"(need >= {2*n_donors})")
            continue

        bot_mask = np.zeros(n_te, dtype=bool); bot_mask[order[:n_donors]]  = True
        top_mask = np.zeros(n_te, dtype=bool); top_mask[order[-n_donors:]] = True

        print(f"  [{label}] resid top-{n_donors}: "
              f"{resid[top_mask].min():.1f}–{resid[top_mask].max():.1f} yr  "
              f"bot-{n_donors}: {resid[bot_mask].min():.1f}–{resid[bot_mask].max():.1f} yr")

        diff = shap_percentile_diff(Xte_z, m.coef_, baseline, top_mask, bot_mask)
        order_diff = np.argsort(diff)[::-1]

        for top_n_peaks in TOP_N_PEAKS_LIST:
            top_idx = order_diff[:top_n_peaks]
            write_bed(label, n_donors, top_n_peaks, feat_names, top_idx)


def main() -> None:
    print("Loading PFC-357 all-cells pseudobulk ...")
    with open(PFC_AC_PKL, "rb") as f:
        pfc_ac = pickle.load(f)
    pfc_ages   = pfc_ac["age"].astype(float)
    pfc_matrix = pfc_ac["peak_sum"].astype(np.float32)
    del pfc_ac; gc.collect()

    valid_tr      = ~np.isnan(pfc_ages)
    y_train       = pfc_ages[valid_tr]
    pfc_matrix_ok = pfc_matrix[valid_tr]
    del pfc_matrix; gc.collect()

    print("Loading SEA-AD pseudobulks ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    sea_donors    = np.array(sea_ac["donors"])
    sea_ages      = sea_ac["age"].astype(float)
    sea_matrix    = sea_ac["peak_sum"].astype(np.float32)
    sea_peaks_all = sea_ac["peak_names"]
    del sea_ac; gc.collect()

    man     = pd.read_csv(ROOT / "cache" / "donor_manifest_annot.csv")
    dlpfc43 = set(man[
        (man.tissue == "DLPFC") & (man.modality == "snatac") &
        (man.kind == "fragments") & man.donor_id.notna()
    ]["donor_id"])
    dlpfc_mask   = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_ages   = sea_ages[dlpfc_mask]
    dlpfc_matrix = sea_matrix[dlpfc_mask]
    del sea_matrix; gc.collect()
    print(f"  DLPFC donors: {dlpfc_mask.sum()}")

    print("Loading overlap mapping ...")
    ov            = np.load(OVERLAP_NPZ, allow_pickle=True)
    valid_sea_idx = ov["valid_sea_idx"]
    sea_to_pfc    = list(ov["sea_to_pfc"])
    feat_names    = [sea_peaks_all[i] for i in valid_sea_idx]
    del sea_peaks_all; gc.collect()

    print("Building PFC all-cells overlap matrix ...")
    Xtr_ac = build_pfc_overlap_matrix(pfc_matrix_ok, sea_to_pfc, len(y_train))
    del pfc_matrix_ok; gc.collect()
    Xte_ac = dlpfc_matrix[:, valid_sea_idx]

    # ── all_cells ─────────────────────────────────────────────────────────────
    print("\n=== all_cells ===")
    Xtr_z_ac = zscore_per_donor(Xtr_ac.astype(np.float64)).astype(np.float32)
    Xte_z_ac = zscore_per_donor(Xte_ac.astype(np.float64)).astype(np.float32)
    del Xtr_ac; gc.collect()

    process_label("all_cells", Xtr_z_ac, y_train, Xte_z_ac, dlpfc_ages, feat_names)
    del Xtr_z_ac, Xte_z_ac; gc.collect()

    # ── per-CT ────────────────────────────────────────────────────────────────
    print("\nLoading PFC per-CT pseudobulks ...")
    with open(PFC_CT_PKL, "rb") as f:
        pfc_ct = pickle.load(f)
    with open(SEA_CT_PKL, "rb") as f:
        sea_ct = pickle.load(f)

    for ct_key, ct_short in CT_MAP.items():
        print(f"\n=== {ct_short} ===")
        pfc_ct_mat = np.array(pfc_ct.pop(ct_key, None), dtype=np.float32)
        sea_ct_mat = np.array(sea_ct.pop(ct_key, None), dtype=np.float32)
        if pfc_ct_mat is None or sea_ct_mat is None:
            print("  skipped — missing key"); continue

        pfc_ct_ok    = pfc_ct_mat[valid_tr]
        sea_ct_dlpfc = sea_ct_mat[dlpfc_mask]
        del pfc_ct_mat, sea_ct_mat; gc.collect()

        tr_nz   = (pfc_ct_ok > 0).any(axis=1)
        te_nz   = (sea_ct_dlpfc > 0).any(axis=1)
        Xtr_ct  = pfc_ct_ok[tr_nz]
        Xte_ct  = sea_ct_dlpfc[te_nz]
        y_tr_ct = y_train[tr_nz]
        ages_te = dlpfc_ages[te_nz]
        del pfc_ct_ok, sea_ct_dlpfc; gc.collect()

        if len(Xtr_ct) < 10:
            print(f"  skipped — too few training donors ({len(Xtr_ct)})"); continue

        Xtr_z = zscore_per_donor(Xtr_ct.astype(np.float64)).astype(np.float32)
        Xte_z = zscore_per_donor(Xte_ct.astype(np.float64)).astype(np.float32)
        del Xtr_ct, Xte_ct; gc.collect()

        process_label(ct_short, Xtr_z, y_tr_ct, Xte_z, ages_te, feat_names)
        del Xtr_z, Xte_z; gc.collect()

    del pfc_ct, sea_ct; gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
