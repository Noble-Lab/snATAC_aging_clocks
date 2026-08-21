#!~/micromamba/envs/sat/bin/python
"""
GREAT and Reactome enrichment of genotype-contrastive age-predictive peaks
in the PFC->SIRT6-OE mouse ATAC clock.

Peaks are ranked by diff = sirt6_mean|SHAP| - wt_mean|SHAP| (the same score
used in pfc_sirt6_shap_pct_dumbbell_labeled_mm10.py):
  - top N by diff (most positive)  = peaks whose age-predictive importance is
    specifically elevated in SIRT6-OE donors relative to WT
  - top N by -diff (most negative) = peaks specifically elevated in WT
Ranking by this signed difference (rather than each genotype's own
mean|SHAP|) makes the two peak sets disjoint by construction, giving a
genuine genotype contrast.

Runs GREAT (GO Biological Process + Mouse Phenotype, matched and
whole-genome background) and Reactome enrichment (via Enrichr, on the
GREAT-derived gene sets) on each peak set, at N = (100, 500, 2000).

External large-file dependency (kept at a fixed absolute path, not inside
this repo): ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
(238MB). Everything else (cache_v4/, figures_v4/, results_v4/, logs/) is
local to this script's own directory.
"""

import logging
import pickle
import re
import sys
import time
from io import StringIO
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

ROOT    = Path(__file__).resolve().parent
FIG     = ROOT / "figures_v4"
RESULTS = ROOT / "results_v4" / "great"
LOG_DIR = ROOT / "logs"
CACHE   = ROOT / "cache_v4"
RESULTS.mkdir(exist_ok=True, parents=True)

ATAC_META_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
PEAK_MAT_PKL  = CACHE / "gse294103_peak_matrices.pkl"
SHAP_PKL      = CACHE / "gse294103_shap_values.pkl"

N_TOP_LIST = (100, 500, 2000)
GROUPS = ("sirt6_specific", "wt_specific")
GROUP_LABEL = {"sirt6_specific": "SIRT6-specific (Δ SHAP)", "wt_specific": "WT-specific (Δ SHAP)"}
GROUP_COLOR = {"sirt6_specific": "#762A83", "wt_specific": "#D6604D"}

GREAT_SUBMIT  = "http://great.stanford.edu/public/cgi-bin/greatWeb.php"
GREAT_TSV     = "http://great.stanford.edu/public/cgi-bin/downloadAllTSV.php"
GREAT_DETAILS = "http://great.stanford.edu/public/cgi-bin/showAllDetails.php"
GENOME_ASSEMBLY_DIR = "/webapp/great/public-4.0.4-ontologies/hg38/20171029"
ONTOLOGIES = [
    "GOBiologicalProcess", "GOMolecularFunction", "GOCellularComponent",
    "HumanPhenotypeOntology", "MGIPhenotype", "MGIPhenoSingleKO",
]
TSV_HEADER = [
    "Ontology", "ID", "Desc", "BinomRank", "BinomP", "BinomBonfP", "BinomFdrQ",
    "RegionFoldEnrich", "ExpRegions", "ObsRegions", "GenomeFrac", "SetCov",
    "HyperRank", "HyperP", "HyperBonfP", "HyperFdrQ", "GeneFoldEnrich",
    "ExpGenes", "ObsGenes", "TotalGenes", "GeneSetCov", "TermCov", "Regions", "Genes",
]
FORE_NAME = "user-provided data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sirt6_wt_diff_great_reactome.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()


# ── Rank peaks by (SIRT6 - WT) mean|SHAP| difference & write BED files ───────
def make_bed_files():
    with open(PEAK_MAT_PKL, "rb") as f:
        unique_pfc, X_sirt6, sample_cols = pickle.load(f)
    with open(ATAC_META_PKL, "rb") as f:
        meta = pickle.load(f)
    peak_names_full = meta["peak_names"]
    peak_names_used = [peak_names_full[i] for i in unique_pfc]
    with open(SHAP_PKL, "rb") as f:
        shap_vals = pickle.load(f)
    abs_shap = np.abs(shap_vals)

    sirt6_mask = np.array(["sirt6" in s for s in sample_cols])
    wt_mask    = np.array(["sirt6" not in s for s in sample_cols])
    sirt6_mean = abs_shap[sirt6_mask].mean(0)
    wt_mean    = abs_shap[wt_mask].mean(0)
    diff = sirt6_mean - wt_mean   # positive = more important in SIRT6-OE

    log.info(f"n SIRT6 donors={sirt6_mask.sum()}, n WT donors={wt_mask.sum()}")
    log.info(f"diff range: [{diff.min():.4f}, {diff.max():.4f}]")

    def to_row(name):
        ch, coords = name.split(":")
        s, e = coords.split("-")
        return ch, int(s), int(e)

    bg_bed = RESULTS / "background_all_peaks.bed"
    assert bg_bed.exists(), "background BED should already exist from prior runs"

    order_sirt6 = np.argsort(diff)[::-1]   # most positive diff first
    order_wt    = np.argsort(diff)         # most negative diff first

    bed_paths = {}
    orders = {"sirt6_specific": order_sirt6, "wt_specific": order_wt}
    for g, order in orders.items():
        for n_top in N_TOP_LIST:
            p = RESULTS / f"top{n_top}_{g}_age_peaks.bed"
            idx = order[:n_top]
            if not p.exists():
                rows = sorted((to_row(peak_names_used[i]) for i in idx), key=lambda r: (r[0], r[1]))
                p.write_text("\n".join(f"{c}\t{s}\t{e}" for c, s, e in rows) + "\n")
            edge_diff = diff[idx[-1]]
            log.info(f"{g} N={n_top}: edge diff = {edge_diff:+.4f} -> {p.name}")
            bed_paths[(g, n_top)] = p

    # sanity check: the two peak sets must be disjoint (this is the whole point)
    for n_top in N_TOP_LIST:
        a = set(order_sirt6[:n_top].tolist())
        b = set(order_wt[:n_top].tolist())
        log.info(f"N={n_top}: overlap between sirt6_specific and wt_specific = {len(a & b)}")

    return bed_paths, bg_bed


# ── GREAT web submission ──────────────────────────────────────────────────────
def submit_job(session, fg_bed_path, bg_bed_path=None):
    data = {
        "species": "hg38", "rule": "basalPlusExt", "span": "1000.0",
        "upstream": "5.0", "downstream": "1.0", "twoDistance": "1000.0",
        "oneDistance": "1000.0", "includeCuratedRegDoms": "1",
        "fgChoice": "data", "fgData": fg_bed_path.read_text(),
    }
    if bg_bed_path is None:
        data["bgChoice"] = "wholeGenome"
    else:
        data["bgChoice"] = "data"
        data["bgData"] = bg_bed_path.read_text()
    resp = session.post(GREAT_SUBMIT, data=data, timeout=300)
    resp.raise_for_status()
    m = re.search(r'_sessionName\s*=\s*"([^"]+)"', resp.text)
    if not m:
        err = re.search(r"user error.*?<blockquote>(.*?)</blockquote>", resp.text, re.S)
        raise RuntimeError(f"GREAT submission failed: {err.group(1) if err else 'unknown error'}")
    session_name = m.group(1)
    return {"sessionName": session_name, "outputDir": f"/scratch/great/tmp/results/{session_name}.d/"}


def fetch_ontology_tsv(session, job, onto_name):
    data = {
        "outputDir": job["outputDir"], "outputPath": f"/tmp/results/{job['sessionName']}.d/",
        "ontoName": onto_name, "ontoList": onto_name, "binom": "",
        "genomeAssemblyDir": GENOME_ASSEMBLY_DIR, "species": "hg38",
    }
    resp = session.post(GREAT_TSV, data=data, timeout=300)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)), sep="\t", names=TSV_HEADER, header=None)


def fetch_gene_set(session, job):
    params = {"species": "hg38", "sessionName": job["sessionName"], "foreName": FORE_NAME}
    resp = session.get(GREAT_DETAILS, params=params, timeout=300)
    resp.raise_for_status()
    rows = re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", resp.text)
    region_rows = [r for r in rows if "unnamed" in r[0]]
    genes = set()
    for _, col2 in region_rows:
        for g in re.findall(r"doForm\(\d+, '([^']+)', '[^']*'\)", col2):
            if g != "unnamed":
                genes.add(g)
    return genes, len(region_rows)


def run_great_go_pheno_and_genes(bed_paths, bg_bed):
    all_onto_rows = []
    gene_sets = {}

    for (g, n_top), fg_bed in bed_paths.items():
        gene_cache = RESULTS / f"top{n_top}_{g}_great_genes.txt"

        for bg_label, bg_arg in [("matched", bg_bed), ("wholeGenome", None)]:
            cache_p = RESULTS / f"top{n_top}_{g}_great_{bg_label}.csv"
            need_genes = (bg_label == "matched") and not gene_cache.exists()
            if cache_p.exists() and not need_genes:
                log.info(f"Loading cached {cache_p.name}")
                df = pd.read_csv(cache_p)
            else:
                session = requests.Session()
                log.info(f"Submitting GREAT job: group={g} N={n_top} background={bg_label}")
                job = submit_job(session, fg_bed, bg_arg)
                log.info(f"  sessionName={job['sessionName']}")
                dfs = []
                for onto in ONTOLOGIES:
                    d = fetch_ontology_tsv(session, job, onto)
                    log.info(f"  {onto}: {len(d)} rows")
                    if len(d):
                        dfs.append(d)
                    time.sleep(1)
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                df.to_csv(cache_p, index=False)

                if need_genes:
                    genes, n_regions = fetch_gene_set(session, job)
                    log.info(f"  region-gene map: {n_regions} regions -> {len(genes)} unique genes")
                    gene_cache.write_text("\n".join(sorted(genes)) + "\n")
                time.sleep(2)

            df["n_top"] = n_top
            df["group"] = g
            df["background"] = bg_label
            all_onto_rows.append(df)

        gene_sets[(g, n_top)] = set(gene_cache.read_text().split())

    return pd.concat(all_onto_rows, ignore_index=True), gene_sets


# ── Reactome enrichment on GREAT-derived gene sets ────────────────────────────
def run_reactome(gene_sets):
    all_dfs = []
    for (g, n_top), genes in gene_sets.items():
        cache_p = RESULTS / f"top{n_top}_{g}_reactome.csv"
        if cache_p.exists():
            log.info(f"Loading cached {cache_p.name}")
            df = pd.read_csv(cache_p)
        else:
            log.info(f"Reactome enrichment: group={g} N={n_top} ({len(genes)} GREAT-mapped genes)")
            try:
                enr = gp.enrichr(gene_list=sorted(genes), gene_sets="Reactome_2022", outdir=None, verbose=False)
                df = enr.results.copy()
            except Exception as e:
                log.warning(f"Enrichr Reactome_2022 failed ({e}); trying Reactome_2016")
                enr = gp.enrichr(gene_list=sorted(genes), gene_sets="Reactome_2016", outdir=None, verbose=False)
                df = enr.results.copy()
            df = df.sort_values("Adjusted P-value")
            df.to_csv(cache_p, index=False)
        df["n_top"] = n_top
        df["group"] = g
        df["n_genes_input"] = len(genes)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


# ── Dual-criteria significance filter (GREAT-recommended) for GO/Phenotype ───
def significant_terms(df):
    return df[
        (df.RegionFoldEnrich > 1) & (df.GeneFoldEnrich > 1)
        & (df.BinomFdrQ < 0.05) & (df.HyperFdrQ < 0.05)
    ].sort_values(["group", "n_top", "background", "BinomFdrQ"])


# ── Figure: Reactome, SIRT6-specific vs WT-specific side by side, per N ──────
def plot_reactome(reactome_df):
    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=False)

    for row, g in enumerate(GROUPS):
        for col, n_top in enumerate(N_TOP_LIST):
            ax = axes[row, col]
            sub = reactome_df[(reactome_df.group == g) & (reactome_df.n_top == n_top)].copy()
            sub = sub.sort_values("Adjusted P-value").head(10)
            sig = sub[sub["Adjusted P-value"] < 0.05]

            if len(sig) == 0:
                ax.text(0.5, 0.5, "no Reactome pathway\nAdj-P < 0.05",
                        ha="center", va="center", fontsize=9.5, color="0.4", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                sig = sig.iloc[::-1]
                y = np.arange(len(sig))
                vals = -np.log10(sig["Adjusted P-value"].clip(1e-300))
                terms = (
                    sig["Term"].str.replace(r" R-HSA-\d+$", "", regex=True)
                    + " (" + sig["Overlap"].astype(str) + ")"
                )
                ax.barh(y, vals, color=GROUP_COLOR[g], alpha=0.8, edgecolor="0.3", linewidth=0.5)
                ax.set_yticks(y)
                ax.set_yticklabels(terms, fontsize=7.5)
                ax.axvline(-np.log10(0.05), color="tomato", linewidth=1.0, linestyle="--")
                ax.set_xlabel("-log10(Adj. P)", fontsize=8.5)

            ax.set_title(f"{GROUP_LABEL[g]}, top {n_top} peaks", fontsize=9.5)

    fig.suptitle(
        "Reactome enrichment (Enrichr) on GREAT basalPlusExt region→gene sets\n"
        "peaks ranked by (SIRT6 − WT) mean|SHAP| difference — genotype-contrastive, disjoint peak sets\n"
        "(pathway name: intersecting genes / GREAT-mapped gene-set size)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG / f"sirt6_wt_diff_great_reactome_top100_500_2000.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        log.info(f"Saved {p}")
    plt.close(fig)


def main():
    bed_paths, bg_bed = make_bed_files()
    onto_df, gene_sets = run_great_go_pheno_and_genes(bed_paths, bg_bed)
    onto_df.to_csv(RESULTS / "sirt6_wt_diff_great_all_raw.csv", index=False)

    for k, genes in gene_sets.items():
        log.info(f"GREAT gene set {k}: {len(genes)} genes")

    sig = significant_terms(onto_df)
    sig.to_csv(RESULTS / "sirt6_wt_diff_great_significant_terms.csv", index=False)
    log.info(f"Total significant GO/Phenotype terms (dual criterion): {len(sig)}")
    for _, r in sig.iterrows():
        log_test("pfc_to_perturbation", "sirt6_wt_diff_great_reactome_top100_500_2000",
                 analysis=f"GREAT web ({r.get('Ontology')}) on SIRT6 WT-diff peaks",
                 test="binomial_region_enrichment", group_a=r.get("group"), group_b=r.get("Desc"),
                 p_value=r.get("BinomP"), q_value=r.get("BinomFdrQ"),
                 notes=f"n_top={r.get('n_top')}; background={r.get('background')}")
    for (g, n_top, bg), grp in sig.groupby(["group", "n_top", "background"]):
        log.info(f"group={g} N={n_top} bg={bg}: {len(grp)} significant terms")

    reactome_df = run_reactome(gene_sets)
    reactome_df.to_csv(RESULTS / "sirt6_wt_diff_reactome_all_raw.csv", index=False)
    for _, r in reactome_df.iterrows():
        log_test("pfc_to_perturbation", "sirt6_wt_diff_great_reactome_top100_500_2000",
                 analysis=f"Enrichr (Reactome) on GREAT-derived gene set (group={r.get('group')}, n_top={r.get('n_top')})",
                 test="enrichr_fisher_exact", group_a=r.get("group"), group_b=r.get("Term"),
                 statistic=r.get("Odds Ratio"), p_value=r.get("P-value"), q_value=r.get("Adjusted P-value"),
                 notes=f"n_top={r.get('n_top')}")
    n_sig_reactome = (reactome_df["Adjusted P-value"] < 0.05).sum()
    log.info(f"Total significant Reactome pathways (Adj-P<0.05) across all group/N: {n_sig_reactome}")
    for (g, n_top), grp in reactome_df.groupby(["group", "n_top"]):
        nsig = (grp["Adjusted P-value"] < 0.05).sum()
        log.info(f"group={g} N={n_top}: {nsig} significant Reactome pathways / {len(grp)} tested")

    plot_reactome(reactome_df)
    log.info("Done.")


if __name__ == "__main__":
    main()
