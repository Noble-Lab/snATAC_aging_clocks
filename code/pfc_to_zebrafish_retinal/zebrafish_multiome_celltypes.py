"""
Zebrafish Retinal Multiome (RNA + ATAC) Cell Typing
=====================================================
GSE325620: 14-sample paired snRNA+snATAC 10x multiome.

1. Extract per-sample RNA (32,520 genes) + ATAC (peaks -> danRer11 50kb tiles,
   same tile grid as the PFC per-cell-type reference) matrices from the
   Cell Ranger ARC filtered_feature_bc_matrix.h5 files packed in the tar.
2. Merge RNA -> normalize / HVG / PCA / neighbors / Leiden cluster.
3. Score clusters against curated + *verified-present* zebrafish retina marker
   genes (ct_markers.txt cross-checked against literature) -> annotate cell_type.
4. Direct PFC-CT mapping: correlate each cluster's mean RNA profile (shared
   ortholog gene symbols) against PFC per-cell-type centroid profiles
   -> direct_pfc_ct label, entirely independent of the retinal marker identity.
   PFC reference = human prefrontal cortex 10x Multiome, 357 donors (HBCC+NABEC
   cohorts, Zenodo 18394349).
5. Diagnostic plots: UMAP, marker heatmap, composition, direct-mapping heatmap.

Cell type labels are propagated to the paired ATAC channel via shared barcodes
(true multiome pairing -- no cross-modality integration needed).

Downstream clock training/testing happens in zebrafish_allcell_clock_per_ct.py

External large-file dependencies (kept at fixed absolute paths, not inside
this repo):
  ~/pfc_to_zebrafish_retinal/GSE325620_RAW.tar (1.5GB raw data)
  ~/pfc_to_zebrafish_retinal/cache (this project's own cache/,
    holds several 100MB-3GB intermediates: per-sample bc_matrices, combined
    clustered AnnData objects)
  ~/age_accel_per_cell_type/cache/v3/pfc_pseudobulks_by_ct_v3.pkl
  ~/atac_processing_techniques/cache/pfc_peak_pseudobulk{,_by_ct_native}.pkl
Everything else (figures/, results/, logs/) is local to this script's own
directory.

Run with:
    ~/micromamba/envs/reprod/bin/python3 \
        zebrafish_multiome_celltypes.py
"""

import gc
import logging
import pickle
import re
import tarfile
import time
from pathlib import Path

import anndata
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from scipy.stats import pearsonr

from np_compat_io import safe_pickle_load

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
CACHE   = Path("~/pfc_to_zebrafish_retinal/cache")  # large caches (100s of MB-GB), kept out of the repo
FIG     = ROOT / "figures" / "multiome"
RESULTS = ROOT / "results" / "multiome"
LOG     = ROOT / "logs"
BC_DIR  = CACHE / "multiome_bc_matrices"

TAR_PATH   = Path("~/pfc_to_zebrafish_retinal/GSE325620_RAW.tar")
CHAIN_GZ   = CACHE / "hg38ToDanRer11.over.chain.gz"
V3_PKL     = Path("~/age_accel_per_cell_type/cache/v3/pfc_pseudobulks_by_ct_v3.pkl")
# Existing validated hg38-peak -> danRer11-50kb-tile liftover (6,377 tiles from
# 521,217 native ATAC peaks). Peak-level liftover retains far more shared
# feature space than lifting generic 50kb-tile midpoints (~679 tiles, tested
# and rejected -- most tile midpoints fall in non-conserved/non-alignable DNA,
# whereas called ATAC peaks are enriched at conserved regulatory elements).
PFC_DANRER11_LIFTOVER_PKL = CACHE / "pfc_to_danrer11_feat_idx.pkl"
PFC_PEAK_ALLCELLS_PKL     = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PFC_PEAK_BY_CT_NATIVE_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk_by_ct_native.pkl")

TILE_SIZE      = 50_000
RIDGE_UNUSED   = None
MIN_GENES_CELL = 200
MAX_PCT_MT     = 10.0
N_HVG          = 3000
N_PCS          = 50
LEIDEN_RES     = 0.6

for d in (CACHE, FIG, RESULTS, LOG, BC_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG / "multiome_celltypes.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

# ── danRer11 chromosome sizes (UCSC chr-prefix) ────────────────────────────────
DANRER11_CHROM_SIZES = {
    "chr1":  59578282,  "chr2":  59640629,  "chr3":  62628489,
    "chr4":  78093715,  "chr5":  72500376,  "chr6":  60270059,
    "chr7":  74282399,  "chr8":  54304671,  "chr9":  56459846,
    "chr10": 45420867,  "chr11": 45484837,  "chr12": 49182954,
    "chr13": 52186027,  "chr14": 52660232,  "chr15": 48040578,
    "chr16": 55266484,  "chr17": 53461100,  "chr18": 51023478,
    "chr19": 48449771,  "chr20": 55201332,  "chr21": 45934066,
    "chr22": 39133080,  "chr23": 46223584,  "chr24": 42172926,
    "chr25": 37502051,
}
ENSEMBL_TO_UCSC = {str(i): f"chr{i}" for i in range(1, 26)}

# GEO-confirmed ages (GSE325620; Control_ReSeq=6mo & IPZF17_blind_A2=22mo
# verified against individual GSM characteristics fields, cross-checked
# against the ATAC-only GSE325621 cohort of the same animals). GSM9626488
# (IPZF31_28mo) is a new biological replicate not present in the ATAC-only run.
SAMPLE_META = {
    "GSM9609631": (18, "18mo_ReSeq"),
    "GSM9609632": (6,  "Control_ReSeq"),
    "GSM9609633": (28, "IPZF16_28mo1_multiome"),
    "GSM9609634": (22, "IPZF17_blind_A2_multiome"),
    "GSM9609635": (1,  "IPZF24_1mo_biorep_multiome"),
    "GSM9626487": (18, "IPZF29_18mo_biorep_multiome"),
    "GSM9626488": (28, "IPZF31_28mo_biorep_multiome"),
    "GSM9626489": (48, "IPZF4yr1_multiome"),
    "GSM9626490": (48, "IPZF4yr2_multiome"),
    "GSM9626491": (12, "zf_12mo_Re"),
    "GSM9626492": (1,  "zf_1mo_Re"),
    "GSM9626493": (30, "zf_30mo_Re"),
    "GSM9626494": (36, "zf_36mo_Re"),
    "GSM9626495": (3,  "zf_3mo"),
}

# ── Zebrafish retinal cell type markers ─────────────────────────────────────────
# Cross-checked against ct_markers.txt + literature (Hoang et al 2020 Science;
# zebrafish retina scRNA-seq atlases). Every gene below was verified present in
# the actual 32,520-gene panel before use (see verify_markers()).
# NOTE on ct_markers.txt: "Muller glia: kdr" is very likely mislabeled -- kdr
# (VEGFR2 paralog) is a canonical *vascular endothelial* marker in zebrafish,
# not glial. We keep kdr/kdrl/fli1a/cdh5/pecam1 as a separate Vascular category
# to check where kdr signal actually lands, and use established glial markers
# (rlbp1a, glula, sox9b, slc1a3a, gfap ...) for Muller glia instead.
MARKERS = {
    "Rod":        ["rho", "gnat1", "nr2e3", "nrl", "pde6a", "pde6b", "rom1b", "saga", "sagb"],
    "Cone":       ["gnat2", "arr3a", "arr3b", "pde6c", "opn1sw1", "opn1sw2",
                   "opn1mw1", "opn1mw2", "opn1lw1", "opn1lw2", "gngt2a", "ntng2a", "thrb"],
    "Bipolar":    ["vsx1", "vsx2", "grm6a", "grm6b", "cabp5a", "cabp5b", "isl1"],
    "Muller":     ["rlbp1a", "rlbp1b", "glula", "glulb", "apoeb", "sox9b", "sox9a",
                   "slc1a3a", "vim", "gfap"],
    "RGC":        ["pou4f2", "pou4f1", "pou4f3", "rbpms2b", "rbpms", "sncga", "sncb",
                   "nefla", "neflb", "isl2a", "elavl3", "gap43"],
    "Amacrine":   ["tfap2a", "tfap2b", "gad1b", "gad2", "slc6a9", "chata", "pax6a"],
    "Horizontal": ["lhx1a", "onecut1", "onecut2", "calb2a", "prox1a"],
    "Microglia":  ["ptprc", "mpeg1.1", "csf1ra", "apoc1", "mfap4"],
    "RPE":        ["rpe65a", "rpe65b", "tyrp1b", "best1", "ttr"],
    "Vascular":   ["kdr", "kdrl", "fli1a", "cdh5", "pecam1"],
}

CT_COLORS = {
    "Rod":        "#2196F3",
    "Cone":       "#FF9800",
    "RGC":        "#F44336",
    "Bipolar":    "#9C27B0",
    "Muller":     "#4CAF50",
    "Amacrine":   "#795548",
    "Horizontal": "#00BCD4",
    "Microglia":  "#FF5722",
    "RPE":        "#E91E63",
    "Vascular":   "#607D8B",
    "Unknown":    "#9E9E9E",
}

PFC_CTS = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
PFC_CT_COLORS = {
    "Excitatory": "#D32F2F", "Inhibitory": "#1976D2", "Oligo": "#7B1FA2",
    "Astro": "#388E3C", "Microglia": "#F57C00", "OPC": "#00838F",
}


def _build_tile_tables():
    offsets, n_tiles = {}, {}
    idx = 0
    for ch in sorted(DANRER11_CHROM_SIZES):
        offsets[ch] = idx
        nt = (DANRER11_CHROM_SIZES[ch] + TILE_SIZE - 1) // TILE_SIZE
        n_tiles[ch] = nt
        idx += nt
    return offsets, n_tiles, idx

CHR_OFFSET, CHR_N_TILES, N_TILES_TOTAL = _build_tile_tables()


def make_feat_to_col(unique_tiles: np.ndarray) -> np.ndarray:
    lut = np.full(N_TILES_TOTAL, -1, dtype=np.int32)
    for ci, gi in enumerate(unique_tiles.tolist()):
        lut[gi] = ci
    return lut


# ── Step 0: load existing validated hg38-peak -> danRer11-tile liftover ───────
def load_pfc_danrer11_tile_grid() -> dict:
    """
    Reuses the peak-level liftover already built for the ATAC-only zebrafish
    clock: 521,217 hg38 PFC ATAC peaks -> danRer11 50kb tiles, 10,510 peaks
    successfully lifted -> 6,377 unique danRer11 tiles. Peak-level liftover
    (curated, conserved regulatory elements) retains ~9x more shared feature
    space than lifting generic 50kb-tile midpoints.
    """
    with open(PFC_DANRER11_LIFTOVER_PKL, "rb") as f:
        lo_idx = pickle.load(f)
    unique_tiles = lo_idx["unique_tiles"]          # (6377,) global danRer11 tile idx
    feat_to_col  = make_feat_to_col(unique_tiles)  # LUT: global tile idx -> column
    log.info(f"Loaded PFC->danRer11 peak-level liftover: {len(unique_tiles):,} "
             f"shared tiles (from {len(lo_idx['pfc_cols']):,}/521,217 lifted peaks)")
    return {"unique_tiles": unique_tiles, "feat_to_col": feat_to_col,
            "pfc_cols": lo_idx["pfc_cols"], "inverse": lo_idx["inverse"]}


# ── Step 1: extract per-sample RNA + ATAC(tile) matrices ──────────────────────
def extract_sample_h5(gsm: str, member, tf: tarfile.TarFile) -> dict:
    tmp = Path(f"/tmp/zf_multiome_{gsm}.h5")
    with tf.extractfile(member) as src, open(tmp, "wb") as dst:
        dst.write(src.read())

    with h5py.File(tmp, "r") as f:
        shape = f["matrix"]["shape"][:]           # (n_features, n_barcodes)
        n_feat, n_cells = int(shape[0]), int(shape[1])
        data    = f["matrix"]["data"][:]
        indices = f["matrix"]["indices"][:]
        indptr  = f["matrix"]["indptr"][:]
        ft      = f["matrix"]["features"]["feature_type"][:]
        names   = f["matrix"]["features"]["name"][:]
        interval = f["matrix"]["features"]["interval"][:]
        barcodes = [b.decode() for b in f["matrix"]["barcodes"][:]]

    tmp.unlink(missing_ok=True)

    # CSC (features x cells) -> transpose -> CSR (cells x features)
    X = sp.csc_matrix((data, indices, indptr), shape=(n_feat, n_cells)).T.tocsr()

    is_gex  = ft == b"Gene Expression"
    is_peak = ft == b"Peaks"

    X_rna   = X[:, is_gex]
    rna_genes = [n.decode() for n in names[is_gex]]

    X_peak  = X[:, is_peak]
    peak_intervals = [n.decode() for n in interval[is_peak]]

    return {
        "X_rna": X_rna, "rna_genes": rna_genes,
        "X_peak": X_peak, "peak_intervals": peak_intervals,
        "barcodes": barcodes,
    }


def bin_peaks_to_danrer11_tiles(X_peak: sp.csr_matrix, peak_intervals: list,
                                 feat_to_col: np.ndarray, n_cols: int) -> sp.csr_matrix:
    iv = pd.Series(peak_intervals)
    split = iv.str.extract(r"^(\w+):(\d+)-(\d+)$")
    chrom = split[0].values
    start = pd.to_numeric(split[1], errors="coerce").values
    end   = pd.to_numeric(split[2], errors="coerce").values
    mid   = ((start + end) // 2)

    n_peaks = len(peak_intervals)
    cols = np.full(n_peaks, -1, dtype=np.int64)
    for uc_ens, uc_name in ENSEMBL_TO_UCSC.items():
        m = (chrom == uc_ens)
        if not m.any() or uc_name not in CHR_OFFSET:
            continue
        nt = CHR_N_TILES[uc_name]
        ti = np.clip(mid[m] // TILE_SIZE, 0, nt - 1).astype(np.int64)
        gi = CHR_OFFSET[uc_name] + ti
        cols[m] = feat_to_col[gi]

    valid  = cols >= 0
    rows_p = np.where(valid)[0]
    cols_p = cols[valid]
    P = sp.csr_matrix((np.ones(len(rows_p), dtype=np.float32), (rows_p, cols_p)),
                       shape=(n_peaks, n_cols))
    return (X_peak @ P).tocsr()


def process_all_samples(feat_to_col: np.ndarray, n_tile_cols: int, force: bool = False):
    gsm_list = sorted(SAMPLE_META.keys())
    all_done = all((BC_DIR / f"{g}_rna.npz").exists() and
                    (BC_DIR / f"{g}_atac_tile.npz").exists() and
                    (BC_DIR / f"{g}_meta.pkl").exists() for g in gsm_list)
    if all_done and not force:
        log.info("All per-sample multiome matrices already cached → skip extraction")
        return

    log.info(f"Opening TAR {TAR_PATH} …")
    with tarfile.open(TAR_PATH, "r") as tf:
        members_map = {}
        for m in tf.getmembers():
            if not m.name.endswith(".h5"):
                continue
            gsm = re.match(r"(GSM\d+)_", m.name)
            if gsm:
                members_map[gsm.group(1)] = m

        for gsm in gsm_list:
            rna_p  = BC_DIR / f"{gsm}_rna.npz"
            atac_p = BC_DIR / f"{gsm}_atac_tile.npz"
            meta_p = BC_DIR / f"{gsm}_meta.pkl"
            if rna_p.exists() and atac_p.exists() and meta_p.exists() and not force:
                log.info(f"{gsm}: already cached → skip")
                continue

            age_mo, label = SAMPLE_META[gsm]
            log.info(f"\n{'='*60}\nProcessing {gsm} ({label}, {age_mo}mo) …")
            t0 = time.time()
            d = extract_sample_h5(gsm, members_map[gsm], tf)

            X_tiles = bin_peaks_to_danrer11_tiles(
                d["X_peak"], d["peak_intervals"], feat_to_col, n_tile_cols)

            save_npz(str(rna_p), d["X_rna"])
            save_npz(str(atac_p), X_tiles)
            with open(meta_p, "wb") as f:
                pickle.dump({
                    "barcodes": d["barcodes"], "rna_genes": d["rna_genes"],
                    "gsm": gsm, "age_mo": age_mo, "label": label,
                    "n_peaks_sample": len(d["peak_intervals"]),
                }, f)
            log.info(f"  {gsm}: RNA {d['X_rna'].shape}, ATAC-tile {X_tiles.shape} "
                      f"| {time.time()-t0:.0f}s")
            del d, X_tiles; gc.collect()


# ── Step 2: merge RNA -> AnnData ───────────────────────────────────────────────
def load_combined_rna_adata() -> anndata.AnnData:
    cp = BC_DIR / "combined_rna.h5ad"
    if cp.exists():
        log.info("Loading cached combined RNA AnnData …")
        return anndata.read_h5ad(cp)

    gsm_list = sorted(SAMPLE_META.keys())
    mats, obs_rows, rna_genes_ref = [], [], None

    for gsm in gsm_list:
        rna_p  = BC_DIR / f"{gsm}_rna.npz"
        meta_p = BC_DIR / f"{gsm}_meta.pkl"
        mat = load_npz(str(rna_p))
        with open(meta_p, "rb") as f:
            meta = pickle.load(f)
        if rna_genes_ref is None:
            rna_genes_ref = meta["rna_genes"]
        elif meta["rna_genes"] != rna_genes_ref:
            raise ValueError(f"{gsm}: RNA gene panel differs from reference")

        age_mo, label = SAMPLE_META[gsm]
        for bc in meta["barcodes"]:
            obs_rows.append({"cell_id": f"{gsm}_{bc}", "gsm": gsm, "label": label,
                              "age_mo": age_mo, "barcode": bc})
        mats.append(mat)
        log.info(f"  {gsm}: {mat.shape[0]:,} cells loaded")

    X = sp.vstack(mats, format="csr")
    obs = pd.DataFrame(obs_rows)
    obs.index = obs["cell_id"].values
    var = pd.DataFrame(index=rna_genes_ref)

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
    log.info(f"Combined RNA AnnData: {adata.shape}")
    adata.write_h5ad(cp, compression="gzip")
    return adata


def qc_and_normalize(adata: anndata.AnnData) -> anndata.AnnData:
    if "log1p_norm" in adata.layers:
        log.info("QC/normalize already done → skip")
        return adata

    n0 = adata.shape[0]
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    keep = (adata.obs["n_genes_by_counts"] >= MIN_GENES_CELL) & \
           (adata.obs["pct_counts_mt"] < MAX_PCT_MT)
    adata = adata[keep].copy()
    log.info(f"QC: kept {adata.shape[0]:,}/{n0:,} cells "
             f"(>= {MIN_GENES_CELL} genes, < {MAX_PCT_MT}% mito)")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    return adata


def cluster_rna(adata: anndata.AnnData) -> anndata.AnnData:
    if "leiden" in adata.obs.columns:
        log.info("Leiden clustering already done → skip")
        return adata

    log.info(f"HVG selection (top {N_HVG}, batch-aware) …")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key="gsm",
                                 flavor="seurat")
    adata.raw = adata
    hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=N_PCS, svd_solver="arpack", random_state=42)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]

    log.info(f"Neighbors + Leiden (res={LEIDEN_RES}) …")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15, n_pcs=N_PCS)
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(adata, min_dist=0.3)
    log.info(f"Leiden: {adata.obs['leiden'].nunique()} clusters, "
             f"{adata.shape[0]:,} cells")
    return adata


# ── Step 3: marker verification + cluster annotation ──────────────────────────
def verify_markers(gene_panel: set) -> dict:
    verified = {}
    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in gene_panel]
        missing = [g for g in genes if g not in gene_panel]
        if missing:
            log.warning(f"  {ct}: missing from panel -> {missing}")
        verified[ct] = present
        log.info(f"  {ct}: {len(present)}/{len(genes)} markers verified present")
    return verified


def score_and_annotate(adata: anndata.AnnData, verified_markers: dict):
    clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    all_genes = list(dict.fromkeys(g for gs in verified_markers.values() for g in gs))
    gene_idx = {g: i for i, g in enumerate(adata.raw.var_names)}
    Xraw = adata.raw.X  # log1p-normalized, all genes

    scores = {}
    for cl in clusters:
        mask = (adata.obs["leiden"] == cl).values
        row = {}
        for g in all_genes:
            if g not in gene_idx:
                continue
            col = gene_idx[g]
            row[g] = float(np.asarray(Xraw[mask, col].todense()).mean())
        scores[cl] = row
    score_df = pd.DataFrame(scores).T
    score_df.index.name = "leiden"

    cluster_labels, cluster_ct_scores = {}, {}
    for cl in score_df.index:
        ct_scores = {}
        for ct, genes in verified_markers.items():
            present = [g for g in genes if g in score_df.columns]
            # z-score each marker across clusters before averaging so that
            # highly-expressed-everywhere genes don't dominate the CT score
            if present:
                sub = score_df[present]
                z = (sub - sub.mean()) / (sub.std() + 1e-8)
                ct_scores[ct] = z.loc[cl].mean()
            else:
                ct_scores[ct] = -np.inf
        best_ct = max(ct_scores, key=ct_scores.get)
        cluster_labels[str(cl)] = best_ct
        cluster_ct_scores[str(cl)] = ct_scores
        n_cl = int((adata.obs["leiden"] == cl).sum())
        top3 = sorted(ct_scores.items(), key=lambda kv: -kv[1])[:3]
        log.info(f"  Cluster {cl} (n={n_cl:,}): {best_ct}  "
                 f"[{', '.join(f'{k}={v:.2f}' for k,v in top3)}]")

    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).fillna("Unknown")
    return adata, score_df, cluster_labels, cluster_ct_scores


# ── Step 4: direct PFC-CT mapping (no name assumptions) ───────────────────────
def build_shared_gene_index(zf_genes: list, pfc_genes: list) -> pd.DataFrame:
    cache_path = CACHE / "rna_shared_gene_index_v2.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # zf_genes has ~700 uppercase-collisions (e.g. real "rho" at one index AND
    # a literal duplicate "RHO" at another -- confirmed empirically, source
    # unclear but consistent with unresolved Ensembl paralog/placeholder
    # entries). A naive {g.upper(): i for ...} dict keeps whichever comes
    # LAST, arbitrarily -- for rhodopsin this silently swapped the real rod
    # photoreceptor gene for an unrelated duplicate record, contaminating the
    # Rod pseudobulk's feature identity. ZFIN convention: curated/confirmed
    # zebrafish gene symbols are lowercase; an ALL-CAPS zebrafish "symbol" is
    # typically a placeholder/unconfirmed entry. On a collision, prefer the
    # non-all-caps original (falls back to last-write-wins if all variants
    # are the same case).
    zf_upper = {}
    for i, g in enumerate(zf_genes):
        key = g.upper()
        is_placeholder = g.isupper() and g.isalpha()
        if key not in zf_upper or not is_placeholder:
            zf_upper[key] = i
    pfc_upper = {}
    for i, g in enumerate(pfc_genes):
        if g.startswith("ENSG"):
            continue
        pfc_upper.setdefault(g.upper(), i)  # keep first occurrence

    shared = sorted(set(zf_upper) & set(pfc_upper))
    df = pd.DataFrame({
        "symbol": shared,
        "zf_idx":  [zf_upper[s] for s in shared],
        "pfc_idx": [pfc_upper[s] for s in shared],
    })
    log.info(f"Shared ortholog gene symbols (direct uppercase match, "
             f"collision-resolved): {len(df):,} / zf={len(zf_genes):,}, pfc={len(pfc_genes):,}")
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def select_pfc_ct_marker_cols(cent_mat: np.ndarray, top_n: int = 300) -> np.ndarray:
    """
    Columns (into the shared-gene array) of genes that are actually
    differential between the 6 PFC cell types: top_n most CT-specific genes
    per CT (centroid minus mean-of-other-centroids), unioned across CTs.
    Using the full shared-gene set for correlation (10k+ genes, mostly
    housekeeping/high-expression genes shared by every cell type) swamps the
    cell-identity signal -- confirmed empirically: it collapsed 45/47 retinal
    clusters (including a marker-unambiguous Microglia cluster) onto a single
    PFC CT. Restricting to each CT's top differential genes fixes this.
    """
    n_ct = cent_mat.shape[0]
    marker_cols = set()
    for i in range(n_ct):
        others = [j for j in range(n_ct) if j != i]
        diff = cent_mat[i] - cent_mat[others].mean(axis=0)
        top = np.argsort(diff)[-top_n:]
        marker_cols.update(top.tolist())
    return np.array(sorted(marker_cols))


def direct_map_clusters_to_pfc_ct(adata: anndata.AnnData, v3: dict,
                                   shared: pd.DataFrame) -> tuple:
    """Correlate each Leiden cluster's mean profile (shared ortholog genes,
    restricted to PFC-CT-differential genes) against PFC per-CT centroid
    profiles. No cell-type-name assumptions."""
    # PFC centroids: mean over donors of cpm+log1p, per CT, restricted to shared genes
    pfc_idx = shared["pfc_idx"].values
    centroids = {}
    for ct in PFC_CTS:
        df = v3["rna"]["per_ct"][ct]                    # (357, 38606) raw counts
        valid = df.notna().all(axis=1)
        X = df[valid].values.astype(np.float64)
        rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        X = np.log1p(X / rs * 1e4)
        centroids[ct] = X[:, pfc_idx].mean(axis=0)       # (n_shared,)
    cent_mat = np.vstack([centroids[ct] for ct in PFC_CTS])  # (6, n_shared)

    marker_cols = select_pfc_ct_marker_cols(cent_mat, top_n=300)
    log.info(f"Direct mapping restricted to {len(marker_cols):,} PFC-CT-differential "
             f"genes (union of top-300 per CT) out of {cent_mat.shape[1]:,} shared genes")
    cent_mat_m = cent_mat[:, marker_cols]

    # Retinal cluster profiles: mean log1p-norm expression, shared+differential genes
    zf_idx = shared["zf_idx"].values[marker_cols]
    Xraw = adata.raw.X  # log1p-normalized full-gene matrix (cells x genes, HVG-agnostic)
    # adata.raw.var_names order == original gene order used to build zf_idx
    clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    clust_mat = np.vstack([
        np.asarray(Xraw[(adata.obs["leiden"] == cl).values][:, zf_idx].mean(axis=0)).ravel()
        for cl in clusters
    ])  # (n_clusters, n_marker_genes)

    # Relative (within-dataset) z-score each gene before correlating: raw
    # cross-dataset magnitude correlation is dominated by scale differences
    # between a human snRNA-seq assay and a zebrafish snRNA-seq assay, not by
    # cell identity (confirmed empirically: it made "Inhibitory" win almost
    # every cluster, including obviously non-neuronal ones like Muller/RPE).
    # Z-scoring each gene within its own dataset (PFC: across the 6 CTs;
    # retina: across the 47 clusters) removes that scale bias and leaves the
    # relative up/down pattern that actually encodes cell identity.
    cent_z  = (cent_mat_m - cent_mat_m.mean(axis=0)) / (cent_mat_m.std(axis=0) + 1e-8)
    clust_z = (clust_mat - clust_mat.mean(axis=0)) / (clust_mat.std(axis=0) + 1e-8)

    # Minimum-correlation confidence floor: a pure argmax always crowns a
    # "winner" even when every PFC CT correlates near zero -- a noise-floor
    # tiebreak, not a real identity call (a sibling project on the mouse
    # ortholog of this dataset hit the identical failure mode and fixed it
    # the same way: below-threshold -> "Unknown", excluded from per-CT
    # pseudobulking rather than force-labeled). MIN_CORR=0.15 sits in a clear
    # gap in this dataset's correlation distribution (dense cluster of
    # 0.02-0.12 "noise" values vs. a 0.16-0.56 tier of confident calls).
    MIN_CORR = 0.15
    corr_records, cluster_direct_ct = [], {}
    for ci, cl in enumerate(clusters):
        corrs = [pearsonr(clust_z[ci], cent_z[i])[0] for i in range(len(PFC_CTS))]
        best_i = int(np.argmax(corrs))
        label = PFC_CTS[best_i] if corrs[best_i] >= MIN_CORR else "Unknown"
        cluster_direct_ct[str(cl)] = label
        corr_records.append({"leiden": cl, **{ct: corrs[i] for i, ct in enumerate(PFC_CTS)}})
        log.info(f"  Cluster {cl}: direct_pfc_ct={label} "
                 f"(best={PFC_CTS[best_i]} r={corrs[best_i]:.3f}; all: " +
                 ", ".join(f"{ct}={corrs[i]:.3f}" for i, ct in enumerate(PFC_CTS)) + ")")

    corr_df = pd.DataFrame(corr_records).set_index("leiden")
    adata.obs["direct_pfc_ct"] = adata.obs["leiden"].map(cluster_direct_ct)
    return adata, corr_df, cluster_direct_ct


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_umap(adata: anndata.AnnData):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ct in sorted(adata.obs["cell_type"].unique()):
        mask = adata.obs["cell_type"] == ct
        emb  = adata.obsm["X_umap"][mask.values]
        axes[0].scatter(emb[:, 0], emb[:, 1], c=CT_COLORS.get(ct, "#999"), s=1.5,
                         alpha=0.4, label=f"{ct} ({mask.sum():,})", rasterized=True)
    axes[0].set_title("Marker-based cell type", fontsize=12)
    axes[0].legend(markerscale=6, fontsize=7, frameon=False, loc="best")
    axes[0].set_xlabel("UMAP1"); axes[0].set_ylabel("UMAP2")

    for ct in PFC_CTS:
        mask = adata.obs["direct_pfc_ct"] == ct
        if mask.sum() == 0:
            continue
        emb = adata.obsm["X_umap"][mask.values]
        axes[1].scatter(emb[:, 0], emb[:, 1], c=PFC_CT_COLORS.get(ct, "#999"), s=1.5,
                         alpha=0.4, label=f"{ct} ({mask.sum():,})", rasterized=True)
    axes[1].set_title("Direct-mapped PFC cell type", fontsize=12)
    axes[1].legend(markerscale=6, fontsize=7, frameon=False, loc="best")
    axes[1].set_xlabel("UMAP1"); axes[1].set_ylabel("UMAP2")

    ages = sorted(adata.obs["age_mo"].unique())
    cmap = plt.cm.plasma
    norm_ = mpl.colors.Normalize(vmin=min(ages), vmax=max(ages))
    for age in ages:
        mask = adata.obs["age_mo"] == age
        emb  = adata.obsm["X_umap"][mask.values]
        axes[2].scatter(emb[:, 0], emb[:, 1], c=[cmap(norm_(age))], s=1.5,
                         alpha=0.4, rasterized=True)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_); sm.set_array([])
    plt.colorbar(sm, ax=axes[2], label="Age (months)")
    axes[2].set_title("Age (months)", fontsize=12)
    axes[2].set_xlabel("UMAP1"); axes[2].set_ylabel("UMAP2")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"umap_multiome_celltypes.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved UMAP figure")


def plot_marker_heatmap(score_df: pd.DataFrame, cluster_labels: dict):
    gene_order, ct_bounds = [], []
    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in score_df.columns]
        if present:
            ct_bounds.append((ct, len(gene_order), len(gene_order) + len(present)))
            gene_order.extend(present)
    if not gene_order:
        log.warning("No marker genes found — skipping heatmap")
        return
    sub = score_df[gene_order]
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-8)
    row_labels = [f"Cl{idx} ({cluster_labels.get(str(idx), '?')})" for idx in sub_norm.index]

    fig, ax = plt.subplots(figsize=(max(14, len(gene_order) * 0.28),
                                     max(5, len(sub_norm) * 0.35)))
    im = ax.imshow(sub_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(gene_order)))
    ax.set_xticklabels(gene_order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for ct, lo_, hi_ in ct_bounds:
        ax.axvline(lo_ - 0.5, color="black", lw=0.8)
        ax.text((lo_ + hi_) / 2 - 0.5, -1.5, ct, ha="center", fontsize=7, rotation=45)
    plt.colorbar(im, ax=ax, label="Normalized mean log1p expression", shrink=0.6)
    ax.set_title("Zebrafish retinal multiome clusters — RNA marker expression")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"marker_heatmap_multiome.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved marker heatmap figure")


def plot_direct_mapping_heatmap(corr_df: pd.DataFrame, cluster_labels: dict):
    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_df) * 0.4)))
    mat = corr_df[PFC_CTS].values
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(PFC_CTS))); ax.set_xticklabels(PFC_CTS, rotation=45, ha="right")
    row_labels = [f"Cl{idx} ({cluster_labels.get(str(idx), '?')})" for idx in corr_df.index]
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        j = int(np.argmax(mat[i]))
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                    edgecolor="gold", lw=2))
    plt.colorbar(im, ax=ax, label="Pearson r (shared ortholog genes)", shrink=0.7)
    ax.set_title("Direct PFC-CT mapping: retinal cluster vs PFC CT centroid\n"
                 "(gold = argmax; independent of marker-based cell_type)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"direct_mapping_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved direct-mapping correlation heatmap")


def plot_mapping_confusion(adata: anndata.AnnData):
    ct_order  = [c for c in CT_COLORS if c in adata.obs["cell_type"].unique()]
    tab = pd.crosstab(adata.obs["cell_type"], adata.obs["direct_pfc_ct"])
    tab = tab.reindex(index=ct_order, columns=PFC_CTS).fillna(0)
    frac = tab.div(tab.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frac.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(PFC_CTS))); ax.set_xticklabels(PFC_CTS, rotation=45, ha="right")
    ax.set_yticks(range(len(ct_order))); ax.set_yticklabels(ct_order)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.values[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Fraction of marker-CT cells", shrink=0.7)
    ax.set_xlabel("Direct-mapped PFC cell type"); ax.set_ylabel("Marker-based retinal cell type")
    ax.set_title("Approach A (name/biology) vs Approach B (direct correlation)\nagreement matrix")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"mapping_approach_agreement.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    tab.to_csv(RESULTS / "mapping_approach_crosstab.csv")
    log.info("Saved mapping agreement (confusion) figure + crosstab")


def plot_ct_composition(adata: anndata.AnnData, col: str, colors: dict, fname: str, order: list):
    obs = adata.obs.copy()
    samples = obs.groupby("gsm").agg(age_mo=("age_mo", "first"),
                                      label=("label", "first")).reset_index()
    samples = samples.sort_values("age_mo")
    ct_order = [c for c in order if c in obs[col].unique()]
    counts = obs.groupby(["gsm", col]).size().unstack(fill_value=0)
    fracs  = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(samples))
    for ct in ct_order:
        if ct not in fracs.columns:
            continue
        vals = fracs.reindex(samples["gsm"].values)[ct].fillna(0).values
        ax.bar(range(len(samples)), vals, bottom=bottom, color=colors.get(ct, "#999"),
               label=ct, width=0.8)
        bottom += vals
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([f"{r.label}\n({int(r.age_mo)}mo)" for _, r in samples.iterrows()],
                        rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of cells")
    ax.set_title(f"{col} composition per sample (ordered by age)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {fname} figure")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("Zebrafish retinal multiome — RNA cell typing + direct PFC-CT mapping")
    log.info("=" * 70)

    log.info("Loading PFC per-CT reference (V3; HBCC+NABEC multiome, 357 donors) …")
    v3 = safe_pickle_load(V3_PKL)
    pfc_rna_genes = v3["rna_gene_names"]
    log.info(f"PFC V3: {len(pfc_rna_genes):,} RNA genes, "
             f"{len(v3['donors'])} donors, CTs={v3['groups']}")

    # Step 0: shared danRer11 ATAC tile grid (validated peak-level liftover)
    lo_idx = load_pfc_danrer11_tile_grid()
    feat_to_col = lo_idx["feat_to_col"]
    n_tile_cols = len(lo_idx["unique_tiles"])
    log.info(f"Shared danRer11 ATAC tile grid: {n_tile_cols:,} tiles")

    # Step 1: extract per-sample matrices
    process_all_samples(feat_to_col, n_tile_cols)

    # Step 2: merge RNA, QC, normalize, cluster
    adata = load_combined_rna_adata()
    adata = qc_and_normalize(adata)
    adata = cluster_rna(adata)
    adata.write_h5ad(BC_DIR / "combined_rna.h5ad", compression="gzip")
    log.info(f"\nCombined RNA: {adata.shape}, samples={adata.obs['gsm'].nunique()}, "
             f"clusters={adata.obs['leiden'].nunique()}")

    # Step 3: marker verification + annotation
    if "cell_type" not in adata.obs.columns:
        gene_panel = set(adata.raw.var_names)
        log.info("Verifying marker genes against actual RNA panel …")
        verified = verify_markers(gene_panel)
        adata, score_df, cluster_labels, cluster_ct_scores = score_and_annotate(adata, verified)
        score_df.to_csv(RESULTS / "cluster_marker_scores.csv")
    else:
        score_df = pd.read_csv(RESULTS / "cluster_marker_scores.csv", index_col=0)
        cluster_labels = adata.obs.groupby("leiden")["cell_type"].first().to_dict()
    log.info(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

    # Step 4: direct PFC-CT mapping
    if "direct_pfc_ct" not in adata.obs.columns:
        shared = build_shared_gene_index(list(adata.raw.var_names), pfc_rna_genes)
        adata, corr_df, cluster_direct_ct = direct_map_clusters_to_pfc_ct(adata, v3, shared)
        corr_df.to_csv(RESULTS / "direct_mapping_correlations.csv")
    else:
        corr_df = pd.read_csv(RESULTS / "direct_mapping_correlations.csv", index_col=0)
    log.info(f"Direct PFC-CT labels: {adata.obs['direct_pfc_ct'].value_counts().to_dict()}")

    adata.write_h5ad(BC_DIR / "combined_rna.h5ad", compression="gzip")

    # Save final obs table for the clock-training script
    obs_out = adata.obs[["gsm", "label", "age_mo", "barcode", "leiden",
                          "cell_type", "direct_pfc_ct"]].copy()
    obs_out.to_csv(CACHE / "retinal_multiome_obs.csv")
    log.info(f"Saved obs table ({len(obs_out):,} cells) → retinal_multiome_obs.csv")

    # Step 5: diagnostic plots
    plot_umap(adata)
    plot_marker_heatmap(score_df, cluster_labels)
    plot_direct_mapping_heatmap(corr_df, cluster_labels)
    plot_mapping_confusion(adata)
    plot_ct_composition(adata, "cell_type", CT_COLORS, "ct_composition_marker", list(CT_COLORS))
    plot_ct_composition(adata, "direct_pfc_ct", PFC_CT_COLORS, "ct_composition_direct", PFC_CTS)

    log.info("\nDone. Next: zebrafish_multiome_ct_clocks.py")


if __name__ == "__main__":
    main()
