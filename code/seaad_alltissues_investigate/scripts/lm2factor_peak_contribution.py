"""
For each peak in the PFC-357 -> SEA-AD DLPFC Ridge clock, test whether its
contribution to predicted age (ridge_coef_p * per-donor z-scored accessibility)
is specifically elevated in High-ADNC female donors, beyond the additive
effects of ADNC and sex.

Two models, both fit for all peaks in one matrix operation (n=43 DLPFC donors):

  1. Interaction test: contrib ~ H + F + H*F
     H = 1 if High ADNC, F = 1 if Female. beta3 (the H*F term) > 0 means the
     peak's contribution is elevated specifically in the High-ADNC-Female group.

  2. Age-adjusted group-elevation test (ANCOVA): contrib ~ age + group, with
     four groups {High x Female, High x Male, Not-High x Female, Not-High x Male}
     and High x Female as reference. For each of the other three groups this
     gives a one-sided p-value for "High x Female > group" after controlling
     for donor age; min_gap is the smallest of the three elevations, and
     pval_iut is the intersection-union test p-value (max of the three
     one-sided p-values) for "High x Female exceeds all three groups".

Outputs per label: results/lm_contrib_peak_rankings_{label}.csv, ranked by
beta_hf (age-adjusted High x Female elevation) and including min_gap/diff_HM/
diff_NHF/diff_NHM/pval_iut, which plot_top_iut_scatter_combined.py reads back
to pick the top peak per cell type.
"""
from __future__ import annotations
import gc
import pickle
import sys
import warnings
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_coef

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
CACHE_DIR   = Path("~/atac_processing_techniques/cache")
GA_CACHE    = Path("~/age_accel_per_cell_type/cache")
RES         = ROOT / "results"
FIG         = ROOT / "figures"
SEAAD_META  = GA_CACHE  / "seaad_pseudobulks_by_ct.pkl"
PFC_AC_PKL  = CACHE_DIR / "pfc_peak_pseudobulk.pkl"
PFC_CT_PKL  = CACHE_DIR / "pfc_peak_pseudobulk_by_ct.pkl"
SEA_AC_PKL  = CACHE_DIR / "seaad_peak_pseudobulk.pkl"
SEA_CT_PKL  = CACHE_DIR / "seaad_peak_pseudobulk_by_ct.pkl"
OVERLAP_NPZ = ROOT / "cache" / "pfc521k_x_seaad218k_overlap.npz"
PCT_CACHE   = ROOT / "cache" / "shap_sex_pct_cache.pkl"
COEF_CACHE  = ROOT / "cache" / "ridge_coef_cache.pkl"

ALPHAS = np.logspace(-2, 6, 30)

CT_MAP = {
    "Excitatory": "Exc", "Inhibitory": "Inh", "Oligo": "Oligo",
    "Astro": "Astro",  "Microglia": "Mic", "OPC": "OPC",
}
CT_KEY = {v: k for k, v in CT_MAP.items()}

ALL_LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]


# ── helpers ────────────────────────────────────────────────────────────────────

def zscore_per_donor(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def build_pfc_overlap_matrix(pfc_mat, sea_to_pfc, n_donors):
    n_feat = len(sea_to_pfc)
    out = np.zeros((n_donors, n_feat), dtype=np.float32)
    for col, tile_idxs in enumerate(sea_to_pfc):
        out[:, col] = pfc_mat[:, tile_idxs].sum(axis=1)
    return out


# ── model ──────────────────────────────────────────────────────────────────────

def fit_contribution_model(contrib, is_high, is_female):
    """Fit contrib ~ H + F + H*F for all P peaks simultaneously.

    contrib : (n, P)   peak contributions (coeff_p * zscore_access_pi)
    Returns
    -------
    beta3   : (P,)     interaction coefficient (is_high:is_female)
    tstat   : (P,)     t-statistic for beta3
    pval    : (P,)     two-sided p-value
    """
    n = contrib.shape[0]
    H  = is_high.astype(np.float64)
    F  = is_female.astype(np.float64)

    X = np.column_stack([np.ones(n), H, F, H * F])  # (n, 4)

    # Single lstsq solve for all P peaks at once
    beta, _, _, _ = np.linalg.lstsq(X, contrib.astype(np.float64), rcond=None)
    # beta : (4, P)

    fitted  = X @ beta                                    # (n, P)
    resid   = contrib.astype(np.float64) - fitted        # (n, P)
    mse     = (resid ** 2).sum(axis=0) / (n - 4)        # (P,)

    XtX_inv = np.linalg.inv(X.T @ X)                    # (4, 4), same for all peaks
    var33   = XtX_inv[3, 3]                              # scalar

    beta3 = beta[3, :]                                   # (P,)
    se    = np.sqrt(np.maximum(mse * var33, 0.0))        # (P,)
    tstat = np.where(se > 1e-14, beta3 / se, 0.0)       # (P,)
    df    = n - 4
    pval  = 2 * scipy_stats.t.sf(np.abs(tstat), df=df)  # (P,)

    return beta3, tstat, pval


# ── age-adjusted HF test ───────────────────────────────────────────────────────

def fit_age_ancova(contrib, true_age, is_hf):
    """ANCOVA: contrib ~ age + is_HF  (parallel-lines model).

    Tests whether the High-ADNC × Female trendline sits above the other
    three groups' common trendline after controlling for donor age.

    H0: beta_HF = 0   (one-sided alternative: beta_HF > 0)

    Returns
    -------
    beta_hf : (P,)   age-adjusted elevation of HF group
    tstat   : (P,)   t-statistic
    pval    : (P,)   one-sided p-value (HF > pooled others)
    """
    valid = ~np.isnan(true_age)
    n_v   = valid.sum()
    X = np.column_stack([
        np.ones(n_v),
        true_age[valid].astype(np.float64),
        is_hf[valid].astype(np.float64),
    ])                                                    # (n_v, 3)
    C = contrib[valid].astype(np.float64)                 # (n_v, P)

    beta, _, _, _ = np.linalg.lstsq(X, C, rcond=None)    # (3, P)

    fitted  = X @ beta                                    # (n_v, P)
    resid   = C - fitted                                  # (n_v, P)
    mse     = (resid ** 2).sum(axis=0) / (n_v - 3)      # (P,)

    XtX_inv = np.linalg.inv(X.T @ X)                     # (3, 3)
    var22   = XtX_inv[2, 2]                               # scalar (is_HF variance)

    beta_hf = beta[2, :]                                  # (P,)
    se      = np.sqrt(np.maximum(mse * var22, 0.0))      # (P,)
    tstat   = np.where(se > 1e-14, beta_hf / se, 0.0)   # (P,)
    df      = n_v - 3
    pval    = scipy_stats.t.sf(tstat, df=df)              # one-sided

    return beta_hf, tstat, pval


def fit_mingap_ancova(contrib, true_age, is_high, is_female):
    """4-group parallel-lines ANCOVA: contrib ~ age + g_HM + g_NHF + g_NHM.

    Reference group: High × Female.  The three dummy coefficients measure
    how much each group sits *below* HF after controlling for age.

    Also computes one-sided p-values for each HF > group comparison and
    the intersection-union test (IUT) p-value = max(p_HM, p_NHF, p_NHM),
    which tests the joint null "HF ≤ at least one group" while controlling
    type I error without additional correction.

    Returns
    -------
    diff_HM, diff_NHF, diff_NHM : (P,)  age-adjusted HF elevation above each group
    min_gap  : (P,)  min of the three diffs (maximise for min-gap ranking)
    pval_HM, pval_NHF, pval_NHM : (P,)  one-sided p-values (HF > group)
    pval_iut : (P,)  IUT p-value = max of the three (rank ascending)
    """
    valid = ~np.isnan(true_age)
    n_v   = valid.sum()

    g_HM  = ( is_high  & ~is_female)[valid].astype(np.float64)
    g_NHF = (~is_high  &  is_female)[valid].astype(np.float64)
    g_NHM = (~is_high  & ~is_female)[valid].astype(np.float64)

    X = np.column_stack([
        np.ones(n_v),
        true_age[valid].astype(np.float64),
        g_HM, g_NHF, g_NHM,
    ])                                                    # (n_v, 5)
    C = contrib[valid].astype(np.float64)                 # (n_v, P)

    beta, _, _, _ = np.linalg.lstsq(X, C, rcond=None)    # (5, P)

    fitted  = X @ beta                                    # (n_v, P)
    resid   = C - fitted                                  # (n_v, P)
    df      = n_v - 5
    mse     = (resid ** 2).sum(axis=0) / df              # (P,)

    XtX_inv = np.linalg.inv(X.T @ X)                     # (5, 5)

    # HF elevation above each group = -beta[k]
    diff_HM  = -beta[2]                                   # (P,)
    diff_NHF = -beta[3]
    diff_NHM = -beta[4]
    min_gap  = np.minimum(np.minimum(diff_HM, diff_NHF), diff_NHM)

    # SE of each diff (= SE of -beta[k] = SE of beta[k])
    se_HM  = np.sqrt(np.maximum(mse * XtX_inv[2, 2], 0.0))
    se_NHF = np.sqrt(np.maximum(mse * XtX_inv[3, 3], 0.0))
    se_NHM = np.sqrt(np.maximum(mse * XtX_inv[4, 4], 0.0))

    t_HM  = np.where(se_HM  > 1e-14, diff_HM  / se_HM,  0.0)
    t_NHF = np.where(se_NHF > 1e-14, diff_NHF / se_NHF, 0.0)
    t_NHM = np.where(se_NHM > 1e-14, diff_NHM / se_NHM, 0.0)

    pval_HM  = scipy_stats.t.sf(t_HM,  df=df)            # one-sided HF > HM
    pval_NHF = scipy_stats.t.sf(t_NHF, df=df)            # one-sided HF > NHF
    pval_NHM = scipy_stats.t.sf(t_NHM, df=df)            # one-sided HF > NHM

    pval_iut = np.maximum(np.maximum(pval_HM, pval_NHF), pval_NHM)  # IUT

    return diff_HM, diff_NHF, diff_NHM, min_gap, pval_HM, pval_NHF, pval_NHM, pval_iut


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # ── shared data ───────────────────────────────────────────────────────────
    print("Loading metadata ...")
    # The donor-metadata cache was pickled under numpy>=2.0; _seaad_meta_compat.py loads
    # a numpy-version-agnostic copy of it instead of unpickling it directly.
    from _seaad_meta_compat import load_seaad_meta
    _meta       = load_seaad_meta()
    adnc_series = _meta["Overall AD neuropathological Change"]
    sex_series  = _meta["Sex"]
    del _meta

    man       = pd.read_csv(ROOT / "cache" / "donor_manifest_annot.csv")
    dlpfc43   = set(man[
        (man.tissue == "DLPFC") & (man.modality == "snatac") &
        (man.kind == "fragments") & man.donor_id.notna()
    ]["donor_id"])

    print("Loading overlap mapping ...")
    ov            = np.load(OVERLAP_NPZ, allow_pickle=True)
    valid_sea_idx = ov["valid_sea_idx"]
    sea_to_pfc    = list(ov["sea_to_pfc"])

    print("Loading feature names / gene annotations ...")
    with open(PCT_CACHE, "rb") as f:
        cache = pickle.load(f)
    feat_names = cache["feat_names"]   # len 197576
    feat_genes = cache["feat_genes"]

    print("Loading SEA-AD pseudobulks ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    sea_donors   = np.array(sea_ac["donors"])
    dlpfc_mask   = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_donors = sea_donors[dlpfc_mask]
    sea_ac_raw   = sea_ac["peak_sum"][dlpfc_mask][:, valid_sea_idx]  # (43, P)
    del sea_ac; gc.collect()

    dlpfc_adnc = np.array([adnc_series.get(d) for d in dlpfc_donors])
    dlpfc_sex  = np.array([sex_series.get(d)  for d in dlpfc_donors])

    with open(SEA_CT_PKL, "rb") as f:
        sea_ct = pickle.load(f)

    # ── Ridge coefficient cache ───────────────────────────────────────────────
    if COEF_CACHE.exists():
        print("Loading Ridge coefficient cache ...")
        with open(COEF_CACHE, "rb") as f:
            coef_cache = pickle.load(f)
    else:
        coef_cache = {}

    need_pfc_ac = "all_cells" not in coef_cache
    need_pfc_ct = any(lb not in coef_cache for lb in ALL_LABELS if lb != "all_cells")

    pfc_ct = None
    Xtr_ac = None

    if need_pfc_ac or need_pfc_ct:
        print("Loading PFC-357 pseudobulks ...")
        with open(PFC_AC_PKL, "rb") as f:
            pfc_ac = pickle.load(f)
        pfc_ages   = pfc_ac["age"].astype(float)
        valid_tr   = ~np.isnan(pfc_ages)
        y_train    = pfc_ages[valid_tr]
        pfc_mat_ok = pfc_ac["peak_sum"].astype(np.float32)[valid_tr]
        del pfc_ac; gc.collect()

        if need_pfc_ac:
            print("  Building all-cells overlap matrix ...")
            Xtr_ac = build_pfc_overlap_matrix(pfc_mat_ok, sea_to_pfc, len(y_train))
            del pfc_mat_ok; gc.collect()
        else:
            del pfc_mat_ok; gc.collect()

    if need_pfc_ct:
        with open(PFC_CT_PKL, "rb") as f:
            pfc_ct = pickle.load(f)

    # ── Pass 1: collect all statistics ────────────────────────────────────────
    all_stats = {}
    for label in ALL_LABELS:
        print(f"\n=== Pass 1: {label} ===")

        if label == "all_cells":
            Xte_raw = sea_ac_raw.copy()
        else:
            Xte_raw = sea_ct[CT_KEY[label]][dlpfc_mask]

        is_high   = (dlpfc_adnc == "High")
        is_female = (dlpfc_sex  == "Female")
        donors    = dlpfc_donors.copy()

        nz        = (Xte_raw > 0).any(axis=1)
        Xte_raw   = Xte_raw[nz]
        is_high   = is_high[nz]
        is_female = is_female[nz]
        donors    = donors[nz]

        res_path = RES / f"xds_pfc357_peaks_{label}_all.csv"
        df_age   = pd.read_csv(res_path).set_index("donor_id")["true_age"]
        true_age = np.array([df_age.get(d, np.nan) for d in donors])

        # — Ridge coefficients (train once, cache) —
        if label not in coef_cache:
            print("  Training Ridge clock ...")
            if label == "all_cells":
                Xtr  = Xtr_ac
                y_tr = y_train
            else:
                pfc_raw = np.array(pfc_ct[CT_KEY[label]], dtype=np.float32)
                tr_nz   = (pfc_raw[valid_tr] > 0).any(axis=1)
                Xtr     = pfc_raw[valid_tr][tr_nz]
                y_tr    = y_train[tr_nz]

            Xtr_z = zscore_per_donor(Xtr.astype(np.float64)).astype(np.float32)
            model = RidgeCV(alphas=ALPHAS).fit(Xtr_z, y_tr)
            coef_cache[label] = model.coef_.astype(np.float32)
            for rank, i in enumerate(np.argsort(np.abs(model.coef_))[::-1][:200]):
                log_coef("seaad_alltissues_investigate", clock_name="dlpfc_xds_pfc357_peaks_ridge_clock",
                         feature=feat_names[i], coefficient=float(model.coef_[i]),
                         modality="ATAC", cell_type=label, rank=rank, model_type="RidgeCV")
            del Xtr_z
            if label != "all_cells":
                del Xtr, pfc_raw
            gc.collect()
            print(f"  Ridge alpha={model.alpha_:.3g}  n_train={len(y_tr)}")

        ridge_coef = coef_cache[label].astype(np.float64)

        Xte_z   = zscore_per_donor(Xte_raw.astype(np.float64))
        contrib  = Xte_z * ridge_coef[None, :]
        del Xte_z, Xte_raw

        n, P = contrib.shape
        print(f"  n_donors={n}  n_peaks={P:,}")
        print(f"  Groups — High×F: {(is_high&is_female).sum()}  "
              f"High×M: {(is_high&~is_female).sum()}  "
              f"NotHigh×F: {(~is_high&is_female).sum()}  "
              f"NotHigh×M: {(~is_high&~is_female).sum()}")

        print("  Fitting contribution model ...", flush=True)
        beta3, tstat_b3, pval_b3 = fit_contribution_model(contrib, is_high, is_female)

        order    = np.argsort(pval_b3)
        rank     = np.empty_like(order); rank[order] = np.arange(1, P + 1)
        pval_adj = np.minimum(1.0, pval_b3 * P / rank)
        pval_adj = np.minimum.accumulate(pval_adj[order][::-1])[::-1][np.argsort(order)]

        print("  Fitting age ANCOVA (contrib ~ age + is_HF) ...", flush=True)
        is_hf = is_high & is_female
        beta_hf, tstat_hf, pval_hf = fit_age_ancova(contrib, true_age, is_hf)

        print("  Fitting 4-group min-gap + IUT ANCOVA ...", flush=True)
        diff_HM, diff_NHF, diff_NHM, min_gap, \
            pval_HM, pval_NHF, pval_NHM, pval_iut = fit_mingap_ancova(
                contrib, true_age, is_high, is_female)

        all_stats[label] = dict(
            beta3=beta3, tstat_b3=tstat_b3, pval_b3=pval_b3, pval_adj_b3=pval_adj,
            beta_hf=beta_hf, tstat_hf=tstat_hf, pval_hf=pval_hf,
            diff_HM=diff_HM, diff_NHF=diff_NHF, diff_NHM=diff_NHM, min_gap=min_gap,
            pval_HM=pval_HM, pval_NHF=pval_NHF, pval_NHM=pval_NHM, pval_iut=pval_iut,
            is_high=is_high, is_female=is_female, true_age=true_age,
            donors=donors, nz=nz,
        )
        del contrib, beta3, tstat_b3, pval_b3, pval_adj
        del beta_hf, tstat_hf, pval_hf
        del diff_HM, diff_NHF, diff_NHM, min_gap, pval_HM, pval_NHF, pval_NHM, pval_iut
        gc.collect()

    # ── Joint BH FDR correction across all labels × all peaks ─────────────────
    print("\nApplying joint BH FDR correction across all peaks and cell types ...")
    all_pval_hf = np.concatenate([all_stats[lb]["pval_hf"] for lb in ALL_LABELS])
    P_total     = len(all_pval_hf)
    ord_j       = np.argsort(all_pval_hf)
    rnk_j       = np.empty_like(ord_j); rnk_j[ord_j] = np.arange(1, P_total + 1)
    padj_joint  = np.minimum(1.0, all_pval_hf * P_total / rnk_j)
    padj_joint  = np.minimum.accumulate(padj_joint[ord_j][::-1])[::-1][np.argsort(ord_j)]
    P_per       = len(all_stats[ALL_LABELS[0]]["pval_hf"])
    for i, lb in enumerate(ALL_LABELS):
        all_stats[lb]["padj_hf_joint"] = padj_joint[i * P_per:(i + 1) * P_per]
    n_sig_joint = (padj_joint < 0.05).sum()
    print(f"  {n_sig_joint} peaks significant at joint FDR < 0.05 "
          f"({P_total:,} tests total)")
    del all_pval_hf, padj_joint
    gc.collect()

    # ── Pass 2: recompute contrib and generate plots / CSVs ───────────────────
    for label in ALL_LABELS:
        print(f"\n=== Pass 2: {label} ===")
        st = all_stats[label]

        if label == "all_cells":
            Xte_raw = sea_ac_raw[st["nz"]]
        else:
            Xte_raw = sea_ct[CT_KEY[label]][dlpfc_mask][st["nz"]]

        ridge_coef = coef_cache[label].astype(np.float64)
        Xte_z      = zscore_per_donor(Xte_raw.astype(np.float64))
        contrib    = Xte_z * ridge_coef[None, :]
        del Xte_z, Xte_raw

        beta3     = st["beta3"]
        tstat_b3  = st["tstat_b3"]
        pval_b3   = st["pval_b3"]
        pval_adj  = st["pval_adj_b3"]
        beta_hf   = st["beta_hf"]
        tstat_hf  = st["tstat_hf"]
        padj_hf   = st["padj_hf_joint"]
        diff_HM   = st["diff_HM"]
        diff_NHF  = st["diff_NHF"]
        diff_NHM  = st["diff_NHM"]
        min_gap   = st["min_gap"]
        pval_HM   = st["pval_HM"]
        pval_NHF  = st["pval_NHF"]
        pval_NHM  = st["pval_NHM"]
        pval_iut  = st["pval_iut"]
        is_high   = st["is_high"]
        is_female = st["is_female"]
        true_age  = st["true_age"]
        donors    = st["donors"]

        n, P = contrib.shape

        # — save ranked CSV (ranked by beta_hf) —
        ranked = np.argsort(beta_hf)[::-1]
        df_out = pd.DataFrame({
            "rank":           np.arange(1, P + 1),
            "peak":           [feat_names[i] for i in ranked],
            "gene":           [feat_genes[i] for i in ranked],
            "beta3":          beta3[ranked],
            "tstat_beta3":    tstat_b3[ranked],
            "pval_beta3":     pval_b3[ranked],
            "padj_bh":        pval_adj[ranked],
            "beta_hf_age":    beta_hf[ranked],
            "tstat_hf_age":   tstat_hf[ranked],
            "pval_hf_age":    st["pval_hf"][ranked],
            "padj_hf_joint":  padj_hf[ranked],
            "diff_HM":        diff_HM[ranked],
            "diff_NHF":       diff_NHF[ranked],
            "diff_NHM":       diff_NHM[ranked],
            "min_gap":        min_gap[ranked],
            "pval_HM":        pval_HM[ranked],
            "pval_NHF":       pval_NHF[ranked],
            "pval_NHM":       pval_NHM[ranked],
            "pval_iut":       pval_iut[ranked],
        })
        csv_out = RES / f"lm_contrib_peak_rankings_{label}.csv"
        df_out.to_csv(csv_out, index=False)
        n_sig = (padj_hf < 0.05).sum()
        n_pos_gap = (min_gap > 0).sum()
        n_iut05   = (pval_iut < 0.05).sum()
        print(f"  saved {csv_out.name}  ({n_sig} peaks joint-FDR<0.05  "
              f"{n_pos_gap} peaks min_gap>0  {n_iut05} peaks IUT p<0.05)")
        print(f"  top-5: {list(df_out['peak'].head())}  genes: {list(df_out['gene'].head())}")

        del contrib
        gc.collect()

    # save updated Ridge cache
    print("\nSaving Ridge coefficient cache ...")
    with open(COEF_CACHE, "wb") as f:
        pickle.dump(coef_cache, f, protocol=4)

    del sea_ct
    if Xtr_ac is not None:
        del Xtr_ac
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
