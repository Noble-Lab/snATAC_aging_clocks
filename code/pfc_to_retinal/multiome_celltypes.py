"""
GSE325478 Multiome (paired RNA+ATAC) Retinal Cell Typing
============================================================
Use RNA -- much more sensitive than ATAC gene-
activity proxies -- to call retinal cell types directly.

Two independent annotations are produced per cluster:
  1. cell_type        - retinal-native identity (Rod/Cone/Bipolar/Muller/RGC/
                         Amacrine/Horizontal/Microglia/Astrocyte/Endothelial/...)
                         via canonical retinal marker panel (verifies + extends
                         ct_markers.txt).
  2. pfc_direct_type  - direct label transfer of PFC broad cell classes
                         (Excitatory/Inhibitory/Oligo/Astro/Microglia/OPC) onto
                         retinal cells using the SAME PFC marker panel used to
                         define those classes in cortex, independent of (1).

Output: obs table (barcode, gsm, age_wk, leiden, cell_type, pfc_direct_type)
used downstream by multiome_allcell_clock_per_ct.py to pseudobulk the paired
ATAC channel.

External large-file dependency (kept at a fixed absolute path, not inside
this repo): this project's cache/ dir, ~/pfc_to_retinal/cache,
holds several 100MB-2.7GB intermediates (clustered RNA h5ads, per-cell tile
matrices). Everything else (figures/, results/, logs/) is local to this
script's own directory.
"""

import gc
import logging
import re
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
np.random.seed(42)
sc.settings.verbosity = 1

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
CACHE   = Path("~/pfc_to_retinal/cache")  # large caches (100s of MB-GB), kept out of the repo
FIG     = ROOT / "figures" / "multiome"
RESULTS = ROOT / "results"
LOG     = ROOT / "logs"
RAW_DIR = Path("~/pfc_to_retinal/multiome_raw")  # raw GEO data (2.0GB), kept out of the repo

for d in (CACHE, FIG, RESULTS, LOG):
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

N_HVG        = 3000
N_PCS        = 50
LEIDEN_RES   = 1.2
MIN_GENES    = 200
MIN_CELLS_GENE = 3

# ── Retinal-native marker panel (verified/extended from ct_markers.txt) ─────
# ct_markers.txt flagged Kdr as a Muller marker -- Kdr (Vegfr2) is actually a
# canonical *endothelial* marker, not Muller glia. Kept here under Endothelial
# instead, and Muller uses canonical glial markers. All other user markers
# (Pou4f2, Lhx1, Tfap2a, Ntng2, Gnat1, Vsx1, Ptprc) retained and supplemented
# with additional canonical markers for a more robust multi-gene signature.
RETINAL_MARKERS = {
    "Rod":         ["Rho", "Nrl", "Nr2e3", "Pde6b", "Cngb1", "Gnat1", "Reep6", "Pde6g"],
    "Cone":        ["Arr3", "Gnat2", "Opn1sw", "Opn1mw", "Pde6c", "Pde6h", "Gnb3", "Ntng2"],
    "Bipolar":     ["Vsx2", "Vsx1", "Grik1", "Grm6", "Cabp5", "Prkca", "Scgn", "Trpm1"],
    "Muller":      ["Rlbp1", "Glul", "Apoe", "Slc1a3", "Sox9", "Aqp4", "Dkk3", "Clu"],
    "RGC":         ["Rbpms", "Pou4f1", "Pou4f2", "Pou4f3", "Sncg", "Nefl", "Nefm", "Slc17a6", "Thy1"],
    "Amacrine":    ["Tfap2a", "Tfap2b", "Gad1", "Gad2", "Slc6a9", "Slc32a1"],
    "Horizontal":  ["Onecut1", "Onecut2", "Lhx1", "Calb1"],
    "Microglia":   ["C1qa", "C1qb", "Cx3cr1", "Csf1r", "P2ry12", "Aif1", "Tmem119", "Ptprc"],
    "Astrocyte":   ["Gfap", "Pax2", "S100b", "Aldh1l1"],
    "Endothelial": ["Cldn5", "Pecam1", "Kdr", "Flt1"],
    "RPE":         ["Rpe65", "Best1", "Ttr"],
    "Pericyte":    ["Pdgfrb", "Rgs5"],
}

# ── PFC broad-class marker panel (for DIRECT label transfer onto retina) ────
# Independent of RETINAL_MARKERS -- these are the canonical cortical
# broad-class markers used to define Excitatory/Inhibitory/Oligo/Astro/
# Microglia/OPC in the PFC training data; scored directly on retinal cells.
PFC_MARKERS = {
    "Excitatory": ["Slc17a7", "Slc17a6", "Satb2"],
    "Inhibitory": ["Gad1", "Gad2", "Slc32a1"],
    "Oligo":      ["Mbp", "Plp1", "Mog", "Mag"],
    "Astro":      ["Aqp4", "Gfap", "Slc1a3", "Gja1"],
    "Microglia":  ["Csf1r", "Cx3cr1", "C1qa", "P2ry12"],
    "OPC":        ["Pdgfra", "Cspg4", "Sox10", "Olig1"],
}

# CVD-safe categorical palette (8-slot, fixed order, ΔE-maximized adjacency);
# the 8 most-populous types get validated hues, rare extras get muted fallbacks
# (always paired with a direct legend, never color-alone-dependent).
CT_COLORS = {
    "Rod": "#2a78d6", "Cone": "#1baf7a", "Bipolar": "#eda100", "Muller": "#008300",
    "RGC": "#4a3aa7", "Amacrine": "#e34948", "Horizontal": "#e87ba4", "Microglia": "#eb6834",
    "Astrocyte": "#7a92a3", "Endothelial": "#a68a5c", "RPE": "#7a9e7a", "Pericyte": "#8a7aa8",
    "Unknown": "#9E9E9E",
}
PFC_COLORS = {
    "Excitatory": "#2a78d6", "Inhibitory": "#1baf7a", "Astro": "#eda100",
    "Microglia": "#008300", "Oligo": "#4a3aa7", "OPC": "#e34948", "Unknown": "#9E9E9E",
}


# ── Sample metadata (parsed from filenames) ─────────────────────────────────
def build_sample_meta() -> dict:
    files = sorted(RAW_DIR.glob("*.h5"))
    meta = {}
    pat = re.compile(r"(GSM\d+)_(IP(\d+)wk\d*)_multiome")
    for fp in files:
        m = pat.match(fp.name)
        if not m:
            log.warning(f"Could not parse filename: {fp.name}")
            continue
        gsm, label, age = m.group(1), m.group(2), int(m.group(3))
        meta[gsm] = {"age_wk": age, "label": label, "path": fp}
    log.info(f"Parsed {len(meta)} samples, ages: "
             f"{sorted(set(v['age_wk'] for v in meta.values()))}")
    return meta


# ── Step 1: load + concatenate RNA across all samples ───────────────────────
def load_rna_combined(sample_meta: dict) -> anndata.AnnData:
    cp = CACHE / "multiome_rna_raw.h5ad"
    if cp.exists():
        log.info(f"Loading cached combined raw RNA AnnData from {cp} …")
        return anndata.read_h5ad(cp)

    adatas = []
    for gsm in sorted(sample_meta.keys()):
        m = sample_meta[gsm]
        log.info(f"  Loading {gsm} ({m['label']}, {m['age_wk']}wk) …")
        a = sc.read_10x_h5(str(m["path"]), gex_only=False)
        a.var_names_make_unique()
        rna = a[:, a.var["feature_types"] == "Gene Expression"].copy()
        del a
        gc.collect()

        # per-cell total ATAC signal (from Peaks features) as a QC covariate,
        # cheap to grab here since the h5 is already open via scanpy's read
        rna.obs_names = [f"{gsm}_{bc}" for bc in rna.obs_names]
        rna.obs["gsm"] = gsm
        rna.obs["label"] = m["label"]
        rna.obs["age_wk"] = m["age_wk"]
        rna.var = rna.var[[]]  # drop feature_types/genome/interval (RNA-irrelevant, ATAC-only cols)
        log.info(f"    {rna.n_obs} cells x {rna.n_vars} genes")
        adatas.append(rna)

    combined = anndata.concat(adatas, join="outer", merge="same", index_unique=None)
    combined.var_names_make_unique()
    log.info(f"Combined raw RNA: {combined.shape}")
    combined.write_h5ad(cp, compression="gzip")
    del adatas
    gc.collect()
    return combined


# ── Step 2: QC, normalize, HVG, PCA, neighbors, leiden, UMAP ────────────────
def process_rna(adata: anndata.AnnData) -> anndata.AnnData:
    cp = CACHE / "multiome_rna_clustered.h5ad"
    if cp.exists():
        log.info(f"Loading cached clustered RNA AnnData from {cp} …")
        return anndata.read_h5ad(cp)

    log.info(f"Starting QC on {adata.shape} …")
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    log.info(f"  Pre-filter: n_genes_by_counts median={adata.obs['n_genes_by_counts'].median():.0f}, "
             f"total_counts median={adata.obs['total_counts'].median():.0f}, "
             f"pct_counts_mt median={adata.obs['pct_counts_mt'].median():.2f}%")

    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_GENE)
    # snRNA: mito transcripts should be near-absent (nuclear prep); high pct_counts_mt
    # flags ambient/cytoplasmic contamination or low-quality nuclei
    adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
    log.info(f"  Post-filter: {adata.shape}")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key="gsm")
    log.info(f"  {int(adata.var['highly_variable'].sum())} HVGs selected")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.pp.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack", random_state=42)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]
    del adata_hvg
    gc.collect()

    log.info("Computing neighbors …")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, random_state=42)
    log.info(f"Leiden clustering (res={LEIDEN_RES}) …")
    sc.tl.leiden(adata, resolution=LEIDEN_RES, flavor="igraph",
                 n_iterations=2, directed=False, random_state=42)
    log.info(f"  {adata.obs['leiden'].nunique()} clusters")
    log.info("Computing UMAP …")
    sc.tl.umap(adata, min_dist=0.3, random_state=42)

    # restore full lognorm matrix as X (pca/scale only touched the HVG copy)
    adata.X = adata.layers["lognorm"].copy()

    adata.write_h5ad(cp, compression="gzip")
    log.info(f"Saved clustered RNA AnnData: {adata.shape} → {cp}")
    return adata


# ── Step 3: marker-based cluster scoring + annotation ────────────────────────
def score_and_annotate(adata: anndata.AnnData, marker_dict: dict,
                        prefix: str) -> tuple:
    """
    sc.tl.score_genes per candidate cell type -> mean score per leiden cluster
    -> argmax assignment. Returns (per-cluster score matrix, cluster->label dict).
    """
    present_cts = {}
    for ct, genes in marker_dict.items():
        genes_present = [g for g in genes if g in adata.var_names]
        if len(genes_present) == 0:
            log.warning(f"  [{prefix}] {ct}: no marker genes found, skipping")
            continue
        if len(genes_present) < len(genes):
            log.info(f"  [{prefix}] {ct}: {len(genes_present)}/{len(genes)} markers found")
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=f"{prefix}_{ct}",
                           use_raw=False, random_state=42)
        present_cts[ct] = genes_present

    score_cols = [f"{prefix}_{ct}" for ct in present_cts]
    cluster_scores = adata.obs.groupby("leiden", observed=True)[score_cols].mean()
    cluster_scores.columns = list(present_cts.keys())

    cluster_labels = {}
    for cl in cluster_scores.index:
        best_ct = cluster_scores.loc[cl].idxmax()
        cluster_labels[cl] = best_ct
        n_cl = int((adata.obs["leiden"] == cl).sum())
        log.info(f"  [{prefix}] Cluster {cl} (n={n_cl}): {best_ct} "
                 f"(score={cluster_scores.loc[cl, best_ct]:.3f})")

    return cluster_scores, cluster_labels


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_umap(adata: anndata.AnnData):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    for ct in sorted(adata.obs["cell_type"].unique()):
        mask = (adata.obs["cell_type"] == ct).values
        emb = adata.obsm["X_umap"][mask]
        ax.scatter(emb[:, 0], emb[:, 1], c=CT_COLORS.get(ct, "#999"), s=1.5,
                   alpha=0.5, label=f"{ct} ({mask.sum()})", rasterized=True)
    ax.set_title("Retinal-native cell type (RNA)", fontsize=12)
    ax.legend(markerscale=6, fontsize=7, loc="best", frameon=False, handlelength=1)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[1]
    for ct in sorted(adata.obs["pfc_direct_type"].unique()):
        mask = (adata.obs["pfc_direct_type"] == ct).values
        emb = adata.obsm["X_umap"][mask]
        ax.scatter(emb[:, 0], emb[:, 1], c=PFC_COLORS.get(ct, "#999"), s=1.5,
                   alpha=0.5, label=f"{ct} ({mask.sum()})", rasterized=True)
    ax.set_title("Direct PFC-type label transfer (RNA)", fontsize=12)
    ax.legend(markerscale=6, fontsize=7, loc="best", frameon=False, handlelength=1)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[2]
    ages = sorted(adata.obs["age_wk"].unique())
    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(vmin=min(ages), vmax=max(ages))
    for age in ages:
        mask = (adata.obs["age_wk"] == age).values
        emb = adata.obsm["X_umap"][mask]
        ax.scatter(emb[:, 0], emb[:, 1], c=[cmap(norm(age))], s=1.5, alpha=0.5,
                   rasterized=True)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Age (weeks)")
    ax.set_title("Age (weeks)", fontsize=12)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"multiome_umap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved UMAP figure")


def plot_marker_heatmap(cluster_scores: pd.DataFrame, cluster_labels: dict,
                         cluster_sizes: pd.Series, name: str):
    sub = cluster_scores.copy()
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-8)
    row_labels = [f"Cl{idx} ({cluster_labels.get(idx,'?')}, n={cluster_sizes.get(idx,0)})"
                  for idx in sub_norm.index]

    fig, ax = plt.subplots(figsize=(max(6, len(sub.columns) * 0.7),
                                    max(4, len(sub_norm) * 0.35)))
    im = ax.imshow(sub_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(sub_norm.shape[0]):
        for j in range(sub_norm.shape[1]):
            ax.text(j, i, f"{sub.values[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if sub_norm.values[i,j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Normalized score (per column)", shrink=0.7)
    ax.set_title(f"{name} marker score by cluster")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"multiome_marker_heatmap_{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {name} marker heatmap")


def plot_ct_composition(adata: anndata.AnnData, col: str, colors: dict, name: str):
    obs = adata.obs.copy()
    samples = obs.groupby("gsm", observed=True).agg(
        age_wk=("age_wk", "first"), label=("label", "first")).reset_index()
    samples = samples.sort_values("age_wk")

    counts = obs.groupby(["gsm", col], observed=True).size().unstack(fill_value=0)
    fracs = counts.div(counts.sum(axis=1), axis=0)
    ct_order = [c for c in sorted(obs[col].unique(), key=lambda c: -fracs[c].mean())]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(len(samples))
    for ct in ct_order:
        vals = fracs.loc[samples["gsm"].values, ct].values
        ax.bar(range(len(samples)), vals, bottom=bottom,
               color=colors.get(ct, "#999"), label=ct, width=0.8)
        bottom += vals
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([f"{r.label}\n({r.age_wk}wk)" for _, r in samples.iterrows()],
                        rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of cells")
    ax.set_title(f"{name} composition per sample (ordered by age)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"multiome_composition_{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {name} composition figure")


def plot_crosstab(adata: anndata.AnnData):
    ct = pd.crosstab(adata.obs["cell_type"], adata.obs["pfc_direct_type"])
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ct_frac.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ct_frac.columns))); ax.set_xticklabels(ct_frac.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct_frac.index))); ax.set_yticklabels(ct_frac.index)
    for i in range(ct_frac.shape[0]):
        for j in range(ct_frac.shape[1]):
            v = ct_frac.values[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Row fraction", shrink=0.7)
    ax.set_xlabel("Direct PFC-type label"); ax.set_ylabel("Retinal-native cell type")
    ax.set_title("Retinal-native CT vs. direct PFC-type transfer\n(row-normalized fraction of cells)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"multiome_crosstab_ct_vs_pfctype.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cell_type x pfc_direct_type crosstab figure")
    ct.to_csv(RESULTS / "multiome_crosstab_ct_vs_pfctype.csv")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("GSE325478 multiome RNA cell typing")
    log.info("=" * 70)

    sample_meta = build_sample_meta()
    adata = load_rna_combined(sample_meta)
    adata = process_rna(adata)
    log.info(f"\nClustered RNA: {adata.shape}, {adata.obs['gsm'].nunique()} samples, "
             f"{adata.obs['leiden'].nunique()} leiden clusters")

    if "cell_type" not in adata.obs.columns:
        log.info("\n--- Scoring retinal-native marker panel ---")
        ret_scores, ret_labels = score_and_annotate(adata, RETINAL_MARKERS, "ret")
        adata.obs["cell_type"] = adata.obs["leiden"].map(ret_labels).astype(str)

        log.info("\n--- Scoring PFC direct-transfer marker panel ---")
        pfc_scores, pfc_labels = score_and_annotate(adata, PFC_MARKERS, "pfc")
        adata.obs["pfc_direct_type"] = adata.obs["leiden"].map(pfc_labels).astype(str)

        cluster_sizes = adata.obs["leiden"].value_counts()
        ret_scores.to_csv(RESULTS / "multiome_cluster_scores_retinal.csv")
        pfc_scores.to_csv(RESULTS / "multiome_cluster_scores_pfctype.csv")

        adata.write_h5ad(CACHE / "multiome_rna_clustered.h5ad", compression="gzip")
        log.info("Saved annotated AnnData")
    else:
        log.info("Cell types already annotated, loading cached scores …")
        ret_scores = pd.read_csv(RESULTS / "multiome_cluster_scores_retinal.csv", index_col=0)
        ret_scores.index = ret_scores.index.astype(str)
        pfc_scores = pd.read_csv(RESULTS / "multiome_cluster_scores_pfctype.csv", index_col=0)
        pfc_scores.index = pfc_scores.index.astype(str)
        cluster_sizes = adata.obs["leiden"].value_counts()
        ret_labels = dict(zip(adata.obs["leiden"], adata.obs["cell_type"]))
        pfc_labels = dict(zip(adata.obs["leiden"], adata.obs["pfc_direct_type"]))

    log.info("\nRetinal-native cell type counts:")
    log.info(adata.obs["cell_type"].value_counts().to_string())
    log.info("\nDirect PFC-type label counts:")
    log.info(adata.obs["pfc_direct_type"].value_counts().to_string())

    # Save obs table for downstream ATAC pseudobulking
    obs_out = adata.obs[["gsm", "label", "age_wk", "leiden", "cell_type", "pfc_direct_type"]].copy()
    obs_out["barcode"] = [i.split("_", 1)[1] for i in adata.obs_names]
    obs_out.index.name = "cell_id"
    obs_out.to_csv(CACHE / "multiome_cell_obs.csv")
    log.info(f"Saved cell obs table: {len(obs_out)} cells → cache/multiome_cell_obs.csv")

    # Plots
    plot_umap(adata)
    plot_marker_heatmap(ret_scores, ret_labels, cluster_sizes, "retinal")
    plot_marker_heatmap(pfc_scores, pfc_labels, cluster_sizes, "pfctype")
    plot_ct_composition(adata, "cell_type", CT_COLORS, "celltype")
    plot_ct_composition(adata, "pfc_direct_type", PFC_COLORS, "pfctype")
    plot_crosstab(adata)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
