"""
Build cache/shap_adnc_cache.pkl for the PFC-357 -> SEA-AD DLPFC all-peaks Ridge
clock: linear SHAP values and peak-vs-ADNC Spearman correlations, for all_cells
and each of 6 broad cell types.

For each label, a Ridge clock is trained on 357 PFC donors and its mean|SHAP|
per peak is computed as |coefficient| x mean|feature - feature_mean| (the
closed-form SHAP value for a linear model). Peaks are also correlated
(Spearman) against ADNC severity in the 43 SEA-AD DLPFC donors with fragment
data. Peak coordinates are annotated with their nearest gene from the
Ensembl77 GTF.

The cache is consumed by great_web_shaptop_export_beds.py and
great_web_shapdiffpct_export_beds.py, which recompute their own top-N gene
lists from it.
"""
from __future__ import annotations

import bisect
import gc
import gzip
import re
import warnings
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("~/atac_processing_techniques/cache")

PFC_AC_PKL  = CACHE_DIR / "pfc_peak_pseudobulk.pkl"
PFC_CT_PKL  = CACHE_DIR / "pfc_peak_pseudobulk_by_ct.pkl"
SEA_AC_PKL  = CACHE_DIR / "seaad_peak_pseudobulk.pkl"
SEA_CT_PKL  = CACHE_DIR / "seaad_peak_pseudobulk_by_ct.pkl"
GTF_GZ      = Path("~/.cache/pyensembl/GRCh38/ensembl77/"
                    "Homo_sapiens.GRCh38.77.gtf.gz")
OVERLAP_NPZ = ROOT / "cache" / "pfc521k_x_seaad218k_overlap.npz"

# ── Constants ──────────────────────────────────────────────────────────────────
ALPHAS      = np.logspace(-2, 6, 30)
ADNC_THRESH = 0.3   # Spearman |r| threshold used to report how many peaks pass

CT_MAP = {
    "Excitatory": "Exc", "Inhibitory": "Inh", "Oligo": "Oligo",
    "Astro": "Astro", "Microglia": "Mic", "OPC": "OPC",
}
CELL_TYPES = list(CT_MAP.keys())
ALL_LABELS = ["all_cells"] + [CT_MAP[ct] for ct in CELL_TYPES]
ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
ADNC_NUM   = {v: i for i, v in enumerate(ADNC_ORDER)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def zscore_per_donor(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def log1p_cpm(X: np.ndarray) -> np.ndarray:
    row_sums = X.astype(np.float32).sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return np.log1p(X / row_sums * 1e6)


def spearman_vec(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman r between each column of X (n×p) and vector y (n,)."""
    # double-argsort ranks each column
    X_r = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float32)
    y_r = stats.rankdata(y).astype(np.float32)
    yc  = y_r - y_r.mean()
    Xc  = X_r - X_r.mean(0)
    num   = Xc.T @ yc
    denom = np.sqrt((Xc**2).sum(0) * float((yc**2).sum())) + 1e-15
    return np.clip(num / denom, -1.0, 1.0)


def linear_shap(Xtr_z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    X_dev = np.abs(Xtr_z - Xtr_z.mean(0))
    return (np.abs(coef) * X_dev.mean(0)).astype(np.float32)


def build_pfc_overlap_matrix(pfc_mat, sea_to_pfc, n_donors):
    n_feat = len(sea_to_pfc)
    out = np.zeros((n_donors, n_feat), dtype=np.float32)
    for col, tile_idxs in enumerate(sea_to_pfc):
        out[:, col] = pfc_mat[:, tile_idxs].sum(axis=1)
    return out


# ── GTF annotation ─────────────────────────────────────────────────────────────

def load_tss_index(gtf_gz: Path) -> dict:
    print("  Parsing GTF ...")
    recs = []
    gene_pat    = re.compile(r'gene_name "([^"]+)"')
    biotype_pat = re.compile(r'gene_biotype "([^"]+)"')
    seen = set()
    with gzip.open(gtf_gz, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            parts = line.split("\t")
            if parts[2] != "gene":
                continue
            bio = biotype_pat.search(parts[8])
            if bio and bio.group(1) not in ("protein_coding", "lincRNA", "antisense"):
                continue
            gm = gene_pat.search(parts[8])
            if not gm:
                continue
            gene = gm.group(1)
            ch   = parts[0]
            if not ch.startswith("chr"):
                ch = "chr" + ch
            s, e, strand = int(parts[3]), int(parts[4]), parts[6]
            tss = s if strand == "+" else e
            key = (ch, gene)
            if key not in seen:
                seen.add(key)
                recs.append((ch, tss, gene))
    by_chrom: dict[str, list] = defaultdict(list)
    for ch, tss, gene in recs:
        by_chrom[ch].append((tss, gene))
    idx = {}
    for ch, entries in by_chrom.items():
        entries.sort()
        idx[ch] = ([e[0] for e in entries], [e[1] for e in entries])
    print(f"    {sum(len(v[0]) for v in idx.values()):,} TSSs loaded")
    return idx


def nearest_gene(chrom: str, mid: int, tss_idx: dict) -> str:
    if chrom not in tss_idx:
        return "?"
    starts, genes = tss_idx[chrom]
    pos = bisect.bisect_left(starts, mid)
    best_gene, best_dist = "?", 500_001
    for k in [pos - 1, pos]:
        if 0 <= k < len(starts):
            d = abs(starts[k] - mid)
            if d < best_dist:
                best_dist = d
                best_gene = genes[k]
    return best_gene


def annotate_peaks(peak_names: list, tss_idx: dict) -> list:
    genes = []
    for p in peak_names:
        try:
            ch, rest = p.split(":", 1)
            s, e = rest.split("-")
            mid = (int(s) + int(e)) // 2
            g = nearest_gene(ch, mid, tss_idx)
        except Exception:
            g = "?"
        genes.append(g)
    return genes


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load shared data ──────────────────────────────────────────────────────
    print("Loading PFC-357 all-cells pseudobulk ...")
    with open(PFC_AC_PKL, "rb") as f:
        pfc_ac = pickle.load(f)
    pfc_ages     = pfc_ac["age"].astype(float)
    pfc_matrix   = pfc_ac["peak_sum"].astype(np.float32)   # (357, 521217)
    del pfc_ac; gc.collect()

    valid_tr      = ~np.isnan(pfc_ages)
    y_train       = pfc_ages[valid_tr]
    pfc_matrix_ok = pfc_matrix[valid_tr]
    del pfc_matrix; gc.collect()
    print(f"  PFC donors (non-NaN age): {len(y_train)}")

    print("\nLoading SEA-AD pseudobulks ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    sea_donors = np.array(sea_ac["donors"])
    sea_ages   = sea_ac["age"].astype(float)
    sea_matrix = sea_ac["peak_sum"].astype(np.float32)  
    sea_peaks  = sea_ac["peak_names"]
    del sea_ac; gc.collect()

    # DLPFC 43-donor mask
    man     = pd.read_csv(ROOT / "cache" / "donor_manifest_annot.csv")
    dlpfc43 = set(man[
        (man.tissue == "DLPFC") & (man.modality == "snatac") &
        (man.kind == "fragments") & man.donor_id.notna()
    ]["donor_id"])
    dlpfc_mask   = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_donors = sea_donors[dlpfc_mask]
    dlpfc_ages   = sea_ages[dlpfc_mask]
    dlpfc_matrix = sea_matrix[dlpfc_mask]        # (43, 218882)
    del sea_matrix; gc.collect()
    print(f"  DLPFC donors: {dlpfc_mask.sum()}")

    # ADNC metadata
    print("\nLoading SEAAD metadata ...")
    # cache/seaad_pseudobulks_by_ct.pkl (the source of this metadata) was
    # pickled under numpy 2.0.2 and cannot be unpickled directly in this
    # numpy-1.26 environment; _seaad_meta_compat.py loads a pre-extracted,
    # numpy-version-agnostic copy of just the metadata columns instead.
    from _seaad_meta_compat import load_seaad_meta
    meta        = load_seaad_meta()
    adnc_series = meta["Overall AD neuropathological Change"]

    dlpfc_adnc     = np.array([adnc_series.get(d, None) for d in dlpfc_donors])
    dlpfc_adnc_num = np.array(
        [ADNC_NUM.get(a, np.nan) if a else np.nan for a in dlpfc_adnc],
        dtype=float,
    )
    valid_adnc_ac = ~np.isnan(dlpfc_adnc_num)
    print(f"  DLPFC donors with ADNC: {valid_adnc_ac.sum()}")

    # Overlap mapping
    print("\nLoading overlap mapping ...")
    ov            = np.load(OVERLAP_NPZ, allow_pickle=True)
    valid_sea_idx = ov["valid_sea_idx"]          # (197576,)
    sea_to_pfc    = list(ov["sea_to_pfc"])
    feat_names    = [sea_peaks[i] for i in valid_sea_idx]
    n_feat        = len(feat_names)
    print(f"  Common features: {n_feat:,}")

    # Build all-cells training matrix
    print("\nBuilding PFC all-cells overlap matrix ...")
    Xtr_ac = build_pfc_overlap_matrix(pfc_matrix_ok, sea_to_pfc, len(y_train))
    del pfc_matrix_ok; gc.collect()
    Xte_ac = dlpfc_matrix[:, valid_sea_idx]      # (43, 197576)

    # GTF annotation for all 197576 features
    print("\nLoading GTF and annotating features ...")
    tss_idx    = load_tss_index(GTF_GZ)
    feat_genes = annotate_peaks(feat_names, tss_idx)
    del tss_idx; gc.collect()
    print(f"  Annotated {sum(g != '?' for g in feat_genes):,} / {n_feat:,} peaks")

    # ── Compute SHAP and ADNC-r for each label ────────────────────────────────
    shap_results:   dict[str, np.ndarray] = {}
    adnc_r_results: dict[str, np.ndarray] = {}

    # All-cells
    print("\n=== all_cells ===")
    Xtr_z_ac    = zscore_per_donor(Xtr_ac.astype(np.float64)).astype(np.float32)
    Xte_lcpm_ac = log1p_cpm(Xte_ac)

    print("  RidgeCV training ...")
    m_ac      = RidgeCV(alphas=ALPHAS).fit(Xtr_z_ac, y_train)
    shap_ac   = linear_shap(Xtr_z_ac, m_ac.coef_)
    shap_results["all_cells"] = shap_ac
    print(f"  alpha={m_ac.alpha_:.2g}  top SHAP={shap_ac.max():.4f}")

    print("  ADNC Spearman r ...")
    y_adnc_v = dlpfc_adnc_num[valid_adnc_ac]
    adnc_r_results["all_cells"] = spearman_vec(
        Xte_lcpm_ac[valid_adnc_ac].astype(np.float32), y_adnc_v
    )
    n_pass = int((np.abs(adnc_r_results["all_cells"]) > ADNC_THRESH).sum())
    print(f"  peaks |ADNC r|>{ADNC_THRESH}: {n_pass}")

    del Xtr_z_ac; gc.collect()

    # Per-CT
    print("\nLoading PFC per-CT pseudobulks ...")
    with open(PFC_CT_PKL, "rb") as f:
        pfc_ct = pickle.load(f)
    print("Loading SEA-AD per-CT pseudobulks ...")
    with open(SEA_CT_PKL, "rb") as f:
        sea_ct = pickle.load(f)

    for ct_key, ct_short in CT_MAP.items():
        print(f"\n=== {ct_short} ===")
        pfc_ct_mat   = np.array(pfc_ct.pop(ct_key, None), dtype=np.float32)
        sea_ct_mat   = np.array(sea_ct.pop(ct_key, None), dtype=np.float32)
        if pfc_ct_mat is None or sea_ct_mat is None:
            print("  skipped — missing key")
            continue

        pfc_ct_ok    = pfc_ct_mat[valid_tr]
        sea_ct_dlpfc = sea_ct_mat[dlpfc_mask]          # (43, 197576)
        del pfc_ct_mat, sea_ct_mat; gc.collect()

        tr_nz = (pfc_ct_ok > 0).any(axis=1)
        te_nz = (sea_ct_dlpfc > 0).any(axis=1)

        Xtr_ct        = pfc_ct_ok[tr_nz]
        Xte_ct        = sea_ct_dlpfc[te_nz]
        y_tr_ct       = y_train[tr_nz]
        adnc_num_ct   = dlpfc_adnc_num[te_nz]
        valid_adnc_ct = ~np.isnan(adnc_num_ct)
        del pfc_ct_ok, sea_ct_dlpfc; gc.collect()

        if len(Xtr_ct) < 10 or len(Xte_ct) < 3:
            print(f"  skipped — too few donors (train={len(Xtr_ct)} test={len(Xte_ct)})")
            continue

        Xtr_z_ct    = zscore_per_donor(Xtr_ct.astype(np.float64)).astype(np.float32)
        Xte_lcpm_ct = log1p_cpm(Xte_ct)
        del Xtr_ct, Xte_ct; gc.collect()

        print(f"  RidgeCV (n_train={len(y_tr_ct)}, n_test={sum(te_nz)}) ...")
        m_ct      = RidgeCV(alphas=ALPHAS).fit(Xtr_z_ct, y_tr_ct)
        shap_ct   = linear_shap(Xtr_z_ct, m_ct.coef_)
        shap_results[ct_short] = shap_ct
        print(f"  alpha={m_ct.alpha_:.2g}  top SHAP={shap_ct.max():.4f}")

        print("  ADNC Spearman r ...")
        if valid_adnc_ct.sum() >= 3:
            adnc_r_ct = spearman_vec(
                Xte_lcpm_ct[valid_adnc_ct].astype(np.float32), adnc_num_ct[valid_adnc_ct]
            )
        else:
            adnc_r_ct = np.zeros(n_feat, dtype=np.float32)
        adnc_r_results[ct_short] = adnc_r_ct
        n_pass = int((np.abs(adnc_r_ct) > ADNC_THRESH).sum())
        print(f"  peaks |ADNC r|>{ADNC_THRESH}: {n_pass}")

        del Xtr_z_ct; gc.collect()

    del pfc_ct, sea_ct; gc.collect()

    # ── Save SHAP + ADNC cache ────────────────────────────────────────────────
    shap_cache_path = ROOT / "cache" / "shap_adnc_cache.pkl"
    print(f"\nSaving SHAP cache → {shap_cache_path}")
    with open(shap_cache_path, "wb") as f:
        pickle.dump({
            "shap_results":   shap_results,
            "adnc_r_results": adnc_r_results,
            "feat_names":     feat_names,
            "ALL_LABELS":     ALL_LABELS,
        }, f, protocol=4)
    print("  Done.")


if __name__ == "__main__":
    main()
