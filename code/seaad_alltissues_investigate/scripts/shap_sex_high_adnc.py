"""
Build cache/shap_sex_pct_cache.pkl: sex-stratified mean|SHAP| percentile ranks
for the PFC-357 -> SEA-AD DLPFC all-peaks Ridge clock, restricted to High-ADNC
donors, for all_cells and each of 6 broad cell types.

For each label: train a Ridge clock on 357 PFC donors, compute per-donor
|SHAP| for the High-ADNC DLPFC test donors (|coefficient| x |feature -
train_mean|), average within each sex, and convert each sex's average to a
percentile rank across peaks. Peak coordinates are annotated with their
nearest gene from the Ensembl77 GTF.

The cache is consumed by lm2factor_peak_contribution.py and
plot_top_iut_scatter_combined.py.
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
from scipy.stats import rankdata
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
PCT_CACHE   = ROOT / "cache" / "shap_sex_pct_cache.pkl"

ALPHAS = np.logspace(-2, 6, 30)

CT_MAP = {
    "Excitatory": "Exc", "Inhibitory": "Inh", "Oligo": "Oligo",
    "Astro": "Astro", "Microglia": "Mic", "OPC": "OPC",
}


def zscore_per_donor(X: np.ndarray) -> np.ndarray:
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


# ── GTF annotation ─────────────────────────────────────────────────────────────

def load_tss_index(gtf_gz: Path) -> dict:
    print("  Parsing GTF ...")
    recs, seen = [], set()
    gene_pat    = re.compile(r'gene_name "([^"]+)"')
    biotype_pat = re.compile(r'gene_biotype "([^"]+)"')
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


# ── Core SHAP-percentile computation ──────────────────────────────────────────

def compute_sex_shap_percentiles(
    Xtr: np.ndarray,
    y_tr: np.ndarray,
    Xte: np.ndarray,
    high_female_mask: np.ndarray,
    high_male_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pct_female, pct_male) arrays of length n_feat."""
    Xtr_z = zscore_per_donor(Xtr.astype(np.float64)).astype(np.float32)
    train_z_mean = Xtr_z.mean(axis=0)

    m = RidgeCV(alphas=ALPHAS).fit(Xtr_z, y_tr)
    abs_coef = np.abs(m.coef_).astype(np.float32)

    Xte_z = zscore_per_donor(Xte.astype(np.float64)).astype(np.float32)

    # abs SHAP per donor: |coef_j| * |x_z_ij - train_mean_j|  →  (n_te, n_feat)
    dev = np.abs(Xte_z - train_z_mean[None, :])          # (n_te, n_feat)
    abs_shap = abs_coef[None, :] * dev                   # broadcast

    def _mean_pct(mask):
        if mask.sum() == 0:
            return np.zeros(abs_shap.shape[1], dtype=np.float32)
        mean_s = abs_shap[mask].mean(axis=0)
        return (rankdata(mean_s) / len(mean_s) * 100).astype(np.float32)

    pct_f = _mean_pct(high_female_mask)
    pct_m = _mean_pct(high_male_mask)
    return pct_f, pct_m


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if PCT_CACHE.exists():
        print(f"Percentile cache already exists at {PCT_CACHE}.")
        return

    # ── Shared data ───────────────────────────────────────────────────────────
    print("Loading PFC-357 all-cells pseudobulk ...")
    with open(PFC_AC_PKL, "rb") as f:
        pfc_ac = pickle.load(f)
    pfc_ages   = pfc_ac["age"].astype(float)
    pfc_matrix = pfc_ac["peak_sum"].astype(np.float32)
    valid_tr   = ~np.isnan(pfc_ages)
    y_train    = pfc_ages[valid_tr]
    pfc_matrix_ok = pfc_matrix[valid_tr]
    del pfc_ac, pfc_matrix; gc.collect()

    print("Loading SEA-AD all-cells pseudobulk ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    sea_donors = np.array(sea_ac["donors"])
    sea_matrix = sea_ac["peak_sum"].astype(np.float32)
    sea_peaks  = sea_ac["peak_names"]
    del sea_ac; gc.collect()

    print("Loading DLPFC 43-donor mask ...")
    man      = pd.read_csv(ROOT / "cache" / "donor_manifest_annot.csv")
    dlpfc43  = set(man[
        (man.tissue == "DLPFC") & (man.modality == "snatac") &
        (man.kind == "fragments") & man.donor_id.notna()
    ]["donor_id"])
    dlpfc_mask   = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_donors = sea_donors[dlpfc_mask]
    dlpfc_matrix = sea_matrix[dlpfc_mask]
    del sea_matrix; gc.collect()

    print("Loading metadata ...")
    # cache/seaad_pseudobulks_by_ct.pkl (the source of this metadata) was
    # pickled under numpy 2.0.2 and cannot be unpickled directly in this
    # numpy-1.26 environment; _seaad_meta_compat.py loads a pre-extracted,
    # numpy-version-agnostic copy of just the metadata columns instead.
    from _seaad_meta_compat import load_seaad_meta
    meta        = load_seaad_meta()
    adnc_series = meta["Overall AD neuropathological Change"]
    sex_series  = meta["Sex"]
    gc.collect()

    dlpfc_adnc = np.array([adnc_series.get(d, None) for d in dlpfc_donors])
    dlpfc_sex  = np.array([sex_series.get(d, None)  for d in dlpfc_donors])

    print("Loading overlap mapping ...")
    ov            = np.load(OVERLAP_NPZ, allow_pickle=True)
    valid_sea_idx = ov["valid_sea_idx"]
    sea_to_pfc    = list(ov["sea_to_pfc"])
    feat_names    = [sea_peaks[i] for i in valid_sea_idx]
    n_feat        = len(feat_names)
    print(f"  Common features: {n_feat:,}")

    print("Building PFC all-cells overlap matrix ...")
    Xtr_ac = build_pfc_overlap_matrix(pfc_matrix_ok, sea_to_pfc, len(y_train))
    del pfc_matrix_ok; gc.collect()
    Xte_ac = dlpfc_matrix[:, valid_sea_idx]

    print("Loading GTF and annotating features ...")
    tss_idx    = load_tss_index(GTF_GZ)
    feat_genes = annotate_peaks(feat_names, tss_idx)
    del tss_idx; gc.collect()
    print(f"  Annotated {sum(g != '?' for g in feat_genes):,} / {n_feat:,} peaks")

    # ── Per-label computation ─────────────────────────────────────────────────
    results: dict[str, tuple] = {}   # label → (pct_female, pct_male)

    # all_cells
    print("\n=== all_cells ===")
    high_f_ac = (dlpfc_adnc == "High") & (dlpfc_sex == "Female")
    high_m_ac = (dlpfc_adnc == "High") & (dlpfc_sex == "Male")
    print(f"  High ADNC — Female: {high_f_ac.sum()}  Male: {high_m_ac.sum()}")
    pct_f, pct_m = compute_sex_shap_percentiles(
        Xtr_ac, y_train, Xte_ac, high_f_ac, high_m_ac
    )
    results["all_cells"] = (pct_f, pct_m)
    del Xtr_ac; gc.collect()

    # per-CT
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

        tr_nz = (pfc_ct_ok > 0).any(axis=1)
        te_nz = (sea_ct_dlpfc > 0).any(axis=1)

        Xtr_ct = pfc_ct_ok[tr_nz]
        Xte_ct = sea_ct_dlpfc[te_nz]
        y_tr_ct = y_train[tr_nz]
        del pfc_ct_ok, sea_ct_dlpfc; gc.collect()

        dlpfc_adnc_te = dlpfc_adnc[te_nz]
        dlpfc_sex_te  = dlpfc_sex[te_nz]
        high_f = (dlpfc_adnc_te == "High") & (dlpfc_sex_te == "Female")
        high_m = (dlpfc_adnc_te == "High") & (dlpfc_sex_te == "Male")
        print(f"  High ADNC — Female: {high_f.sum()}  Male: {high_m.sum()}")

        if len(Xtr_ct) < 10 or high_f.sum() < 2 or high_m.sum() < 2:
            print("  skipped — too few donors"); continue

        pct_f, pct_m = compute_sex_shap_percentiles(
            Xtr_ct, y_tr_ct, Xte_ct, high_f, high_m
        )
        results[ct_short] = (pct_f, pct_m)
        del Xtr_ct, Xte_ct; gc.collect()

    del pfc_ct, sea_ct; gc.collect()

    # ── Cache percentile results ──────────────────────────────────────────────
    print(f"\nSaving percentile cache → {PCT_CACHE}")
    with open(PCT_CACHE, "wb") as f:
        pickle.dump({"results": results, "feat_names": feat_names,
                     "feat_genes": feat_genes}, f, protocol=4)
    print("Done.")


if __name__ == "__main__":
    main()
