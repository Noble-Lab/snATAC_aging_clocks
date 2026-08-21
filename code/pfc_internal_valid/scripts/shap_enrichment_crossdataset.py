"""
Three-part analysis on the all-PFC (357 donors) Ridge age clock:

Part 1 — Linear SHAP:
  Train Ridge on all 357 PFC donors (per-donor z-scored) for all cells + 6 CTs.
  mean|SHAP|[j] = |coef[j]| × mean_i(|X_norm[i,j] − X_norm[:,j].mean()|)
  → grouped horizontal barplot (top 5 peaks per CT / all cells).

Part 2 — Pathway enrichment:
  Map top peaks → nearest protein-coding gene (Ensembl77 TSS, pyranges).
  Top 100 unique genes per CT / all cells → gseapy.enrichr (KEGG_2021_Human +
  GO_Biological_Process_2023). BH q-values corrected across ALL CT × pathway
  tests jointly. → grouped barplot (top 5 pathways per CT / all cells).

Part 3 — Cross-dataset age-peak correlation:
  Pearson r(age, log-CPM_peak) for every PFC peak in:
    PFC (357 donors), Hippocampus (40, already in PFC space),
    Mouse hypothalamus (20, mm10 peaks lifted to hg38 → mapped to PFC peaks).
  Rank peaks by mean r across ≥2 datasets (all 3 if hypo liftover exists).
  → 5-row × 3-col plot (PFC scatter | Hip scatter | Hypo violin) for top 5.

Outputs (all results):
  results/shap_mean_abs.pkl            {label: array(521217,)}
  results/peak_gene_map.csv            peak → nearest gene, distance
  results/pathway_enrichment.csv       full enrichment table with q_global
  results/cross_dataset_corr.csv       peak × {r_pfc, r_hip, r_hypo, mean_r}
  figures/shap_barplot.pdf/.png
  figures/pathway_enrichment.pdf/.png
  figures/cross_dataset_top_peaks.pdf/.png
"""
import gc, pickle, sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _numpy2_compat import load_pickle_compat

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_coef, log_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import gseapy as gp
import pyranges as pr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
ATAC_ROOT = Path("~/atac_processing_techniques/cache")  # 758MB+, external
HIP_CACHE = Path("~/pfc_to_hippocampus/cache")          # 579MB+, external
HYPO_CACHE = ROOT.parent / "pfc_to_hypo" / "cache"                   # small, in-repo
RESULTS   = ROOT / "results"; FIG = ROOT / "figures"
for d in (RESULTS, FIG): d.mkdir(parents=True, exist_ok=True)

GTF_PATH  = Path("~/.cache/pyensembl/GRCh38/ensembl77/"
                 "Homo_sapiens.GRCh38.77.gtf.gz")

ALPHAS     = tuple(np.logspace(-2, 6, 30))
CELL_TYPES = ["Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
ALL_LABELS = ["All cells"] + CELL_TYPES

CT_COLORS = {
    "All cells":  "#444444", "Excitatory": "#E64B35",
    "Inhibitory": "#4DBBD5", "Oligo":      "#00A087",
    "Astro":      "#3C5488", "Microglia":  "#F39B7F",
    "OPC":        "#8491B4",
}

ENRICH_LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023"]
TOP_GENES   = 100   # genes per CT for enrichment
TOP_PEAKS   = 5     # peaks to show in barplot

# ── helpers ────────────────────────────────────────────────────────────────
def cpm_log1p(X):
    d = X.astype(np.float32).sum(1, keepdims=True)
    d = np.where(d < 1, 1., d)
    return np.log1p(X / d * 1e6)

def sample_zscore(X):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd

def norm(X):
    return sample_zscore(cpm_log1p(X.astype(np.float32)))

def pearson_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    return float(pearsonr(a[m], b[m])[0])

def pearson_vec(X, y):
    """Vectorised Pearson r for all columns of X against y. X: (n,p), y: (n,)."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    num = Xc.T @ yc
    denom = np.sqrt((Xc**2).sum(0)) * (np.sqrt((yc**2).sum()) + 1e-12)
    return num / (denom + 1e-12)

def parse_peak(name):
    """'chrX:start-end' or 'chrX-start-end' → (chrom, start, end)."""
    s = str(name)
    if ":" in s:
        chrom, rest = s.split(":", 1)
        start, end = rest.split("-")
    else:
        parts = s.rsplit("-", 2)
        chrom, start, end = parts[0], parts[1], parts[2]
    return chrom, int(start), int(end)

# ── load caches ────────────────────────────────────────────────────────────
print("Loading caches…")
with open(ATAC_ROOT / "pfc_peak_pseudobulk.pkl", "rb") as f:
    pfc_all = pickle.load(f)
# pfc_peak_pseudobulk_by_ct_native.pkl was pickled under numpy>=2.0 by the
# sibling atac_processing_techniques project; this env pins numpy<2.0 (see
# scripts/_numpy2_compat.py for why a plain pickle.load() fails here).
pfc_ct = load_pickle_compat(ATAC_ROOT / "pfc_peak_pseudobulk_by_ct_native.pkl")

peak_names = np.array(pfc_all["peak_names"])   # (521217,)
pfc_age    = pfc_all["age"].astype(np.float64)
n_peaks    = len(peak_names)

print(f"PFC: {len(pfc_age)} donors, {n_peaks:,} peaks")


# ══════════════════════════════════════════════════════════════════════════
# Part 1: Linear SHAP
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Linear SHAP ===")

shap_results = {}   # label → mean_abs_shap (n_peaks,)

for label, raw in ([("All cells", pfc_all["peak_sum"])] +
                   [(ct, pfc_ct[ct]) for ct in CELL_TYPES]):
    print(f"  {label}…", flush=True)
    X = norm(raw)                               # (357, 521217)
    m = RidgeCV(alphas=ALPHAS).fit(X, pfc_age)
    coef = m.coef_                              # (521217,)
    # mean|SHAP| = |coef| × mean|X - X_mean|  (avoids full SHAP matrix)
    X_dev = np.abs(X - X.mean(0))              # (357, 521217)
    mean_abs_shap = np.abs(coef) * X_dev.mean(0)
    shap_results[label] = mean_abs_shap.astype(np.float32)
    print(f"    alpha={m.alpha_:.3g}  top SHAP={mean_abs_shap.max():.4f}")

    # Log top-200 |coefficient| peaks for this Ridge clock (full 521,217-length
    # coefficient vector is impractical as a flat table; top peaks are what
    # figures 1/4/9/11 actually report on).
    top_coef_idx = np.argsort(np.abs(coef))[::-1][:200]
    for rank, idx in enumerate(top_coef_idx):
        log_coef(
            "pfc_internal_valid", clock_name="all_PFC_357donor_ridge_clock",
            feature=peak_names[idx], coefficient=float(coef[idx]),
            modality="ATAC", cell_type=label, rank=rank, model_type="RidgeCV",
        )
    del X, X_dev, coef; gc.collect()

with open(RESULTS / "shap_mean_abs.pkl", "wb") as f:
    pickle.dump(shap_results, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved {RESULTS/'shap_mean_abs.pkl'}")


# ══════════════════════════════════════════════════════════════════════════
# Part 2a: SHAP barplot
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 2a: SHAP barplot ===")

# Build top-5-peaks table for plotting
shap_rows = []
for label in ALL_LABELS:
    s = shap_results[label]
    top_idx = np.argsort(s)[::-1][:TOP_PEAKS]
    for rank, idx in enumerate(top_idx):
        shap_rows.append(dict(
            label=label, rank=rank, peak_idx=idx,
            peak_name=peak_names[idx], shap=float(s[idx]),
        ))
df_shap = pd.DataFrame(shap_rows)

n_labels = len(ALL_LABELS)
n_bars   = TOP_PEAKS
bar_h    = 0.12
group_h  = n_bars * bar_h + 0.3
fig_h    = n_labels * group_h + 0.8

fig, ax = plt.subplots(figsize=(8, fig_h))
y_base = 0.0
yticks, ylabels = [], []

for g_idx, label in enumerate(ALL_LABELS[::-1]):   # bottom = all cells
    sub = df_shap[df_shap.label == label].sort_values("shap")
    color = CT_COLORS[label]
    ys = [y_base + i * bar_h for i in range(len(sub))]
    ax.barh(ys, sub["shap"].values, height=bar_h * 0.8,
            color=color, alpha=0.85, label=label if g_idx == 0 else "")
    ax.text(-0.0005, y_base + (len(sub)-1) * bar_h / 2,
            label, ha="right", va="center", fontsize=7.5,
            fontweight="bold", color=color)
    for y, (_, row) in zip(ys, sub.iterrows()):
        ax.text(row["shap"] + ax.get_xlim()[1]*0.005 if ax.get_xlim()[1] > 0 else 1e-4,
                y, row["peak_name"], va="center", fontsize=5.5, color="#333333")
    yticks.extend(ys); ylabels.extend([""] * len(sub))
    y_base += group_h

ax.set_yticks([]); ax.set_xlabel("Mean |SHAP| value", fontsize=9)
ax.set_title("Top age-predictive peaks per cell type (linear SHAP)", fontsize=10)
ax.tick_params(labelsize=7)
fig.tight_layout()
for ext in ("pdf","png"):
    p = FIG / f"shap_barplot.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150); print(f"Saved {p}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Part 2b: Peak → gene annotation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 2b: Peak → gene annotation ===")

print("  Loading Ensembl77 GTF (protein-coding genes)…")
gtf_full = pr.read_gtf(str(GTF_PATH))
genes    = gtf_full[(gtf_full.Feature == "gene") &
                    (gtf_full.gene_biotype == "protein_coding")]
# TSS: Start for + strand, End for − strand
tss_df   = genes.df[["Chromosome","Start","End","Strand","gene_name"]].copy()
tss_df["tss"] = np.where(tss_df["Strand"]=="+", tss_df["Start"], tss_df["End"])
tss_df["chr"] = "chr" + tss_df["Chromosome"].astype(str)
tss_pr   = pr.from_dict({
    "Chromosome": tss_df["chr"].values,
    "Start":      tss_df["tss"].values,
    "End":        (tss_df["tss"] + 1).values,
    "gene_name":  tss_df["gene_name"].values,
})

# Build peaks PyRanges (use peak midpoints for nearest query)
chroms, starts, ends = [], [], []
for name in peak_names:
    c, s, e = parse_peak(name)
    chroms.append(c); starts.append(s); ends.append(e)
chroms = np.array(chroms); starts = np.array(starts); ends = np.array(ends)
mids   = (starts + ends) // 2
peaks_pr = pr.from_dict({
    "Chromosome": chroms, "Start": mids, "End": mids + 1,
    "peak_idx":   np.arange(n_peaks),
})

print("  Running nearest TSS…")
nearest = peaks_pr.nearest(tss_pr)
near_df = nearest.df[["peak_idx","gene_name","Distance"]].copy()
near_df.columns = ["peak_idx","gene_name","tss_dist"]
near_df["peak_name"] = peak_names[near_df["peak_idx"].values]

near_df.to_csv(RESULTS / "peak_gene_map.csv", index=False)
print(f"  Saved {RESULTS/'peak_gene_map.csv'}  ({len(near_df):,} peaks annotated)")

# Build quick lookup: peak_idx → gene_name
idx2gene = dict(zip(near_df["peak_idx"].values, near_df["gene_name"].values))


# ══════════════════════════════════════════════════════════════════════════
# Part 2c: Pathway enrichment
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 2c: Pathway enrichment ===")

enrich_rows = []
for label in ALL_LABELS:
    s = shap_results[label]
    # top TOP_GENES unique genes by SHAP rank
    sorted_idx = np.argsort(s)[::-1]
    genes_ordered, seen = [], set()
    for idx in sorted_idx:
        g = idx2gene.get(int(idx))
        if g and g not in seen:
            seen.add(g); genes_ordered.append(g)
        if len(genes_ordered) >= TOP_GENES:
            break

    print(f"  {label}: {len(genes_ordered)} genes, running enrichr…", flush=True)
    for lib in ENRICH_LIBS:
        # Enrichr (maayanlab.cloud) rate-limits (HTTP 429) under bursty calls;
        # retry with backoff rather than silently dropping this (label, lib).
        for attempt in range(4):
            try:
                res = gp.enrichr(gene_list=genes_ordered,
                                 gene_sets=lib, outdir=None, verbose=False)
                df = res.results.copy()
                df["label"]   = label
                df["library"] = lib
                enrich_rows.append(df)
                break
            except Exception as e:
                print(f"    enrichr error ({lib}) attempt {attempt+1}/4: {e}")
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))

df_enrich = pd.concat(enrich_rows, ignore_index=True)

# BH correction jointly across all label × library × term combinations
raw_p = df_enrich["P-value"].values.astype(float)
_, q_global, _, _ = multipletests(raw_p, method="fdr_bh")
df_enrich["q_global"] = q_global

df_enrich.to_csv(RESULTS / "pathway_enrichment.csv", index=False)
print(f"Saved {RESULTS/'pathway_enrichment.csv'}  ({len(df_enrich):,} tests)")

for _, r in df_enrich.iterrows():
    log_test(
        "pfc_internal_valid", "shap_barplot / pathway_enrichment",
        analysis=f"Enrichr ({r['library']}) on top-{TOP_GENES} SHAP genes",
        test="enrichr_fisher_exact", group_a=r["label"], group_b=r["Term"],
        statistic=r.get("Odds Ratio"), p_value=r.get("P-value"), q_value=r.get("q_global"),
        notes=f"library={r['library']}",
    )


# ══════════════════════════════════════════════════════════════════════════
# Part 2d: Pathway barplot
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 2d: Pathway barplot ===")

# Keep significant terms (q_global < 0.1) and take top 5 per label
sig  = df_enrich[df_enrich["q_global"] < 0.1].copy()
sig["neglog_q"] = -np.log10(sig["q_global"].clip(1e-300))

top_paths_rows = []
for label in ALL_LABELS:
    sub = sig[sig.label == label].sort_values("q_global").head(TOP_PEAKS)
    for _, row in sub.iterrows():
        top_paths_rows.append(dict(
            label=label, term=row["Term"],
            neglog_q=row["neglog_q"], q_global=row["q_global"],
            library=row["library"],
        ))
df_top_paths = pd.DataFrame(
    top_paths_rows,
    columns=["label", "term", "neglog_q", "q_global", "library"],
)

bar_h = 0.14
fig, ax = plt.subplots(figsize=(9, fig_h))
y_base = 0.0

for g_idx, label in enumerate(ALL_LABELS[::-1]):
    sub = df_top_paths[df_top_paths["label"] == label].sort_values("neglog_q")
    if sub.empty:
        y_base += group_h; continue
    color = CT_COLORS[label]
    ys = [y_base + i * bar_h for i in range(len(sub))]
    ax.barh(ys, sub["neglog_q"].values, height=bar_h * 0.8,
            color=color, alpha=0.85)
    ax.text(-0.05, y_base + (len(sub)-1) * bar_h / 2,
            label, ha="right", va="center", fontsize=7.5,
            fontweight="bold", color=color)
    for y, (_, row) in zip(ys, sub.iterrows()):
        term = str(row["term"])
        term = term[:55] + "…" if len(term) > 55 else term
        ax.text(row["neglog_q"] + 0.03, y, term,
                va="center", fontsize=5.5, color="#333333")
    y_base += group_h

ax.set_yticks([])
ax.set_xlabel("−log₁₀(q) — BH corrected across all cell types × libraries", fontsize=8)
ax.set_title("Top pathway enrichments per cell type (top 100 SHAP genes)", fontsize=10)
fig.tight_layout()
for ext in ("pdf","png"):
    p = FIG / f"pathway_enrichment.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150); print(f"Saved {p}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Part 3: Cross-dataset age-peak correlation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: Cross-dataset correlation ===")

# ── 3a: PFC ───────────────────────────────────────────────────────────────
print("  PFC Pearson…", flush=True)
X_pfc = cpm_log1p(pfc_all["peak_sum"])       # (357, 521217)
r_pfc = pearson_vec(X_pfc.astype(np.float64), pfc_age)  # (521217,)
del X_pfc; gc.collect()

# ── 3b: Hippocampus ───────────────────────────────────────────────────────
print("  Loading hippocampus…", flush=True)
with open(HIP_CACHE / "hip_peak_pseudobulks_in_pfc_space.pkl", "rb") as f:
    hip = pickle.load(f)
hip_age = hip["age"].astype(np.float64)
X_hip   = cpm_log1p(hip["all_cells"]["peak_sum"])   # (40, 521217)
r_hip   = pearson_vec(X_hip.astype(np.float64), hip_age)
del X_hip; gc.collect()

# ── 3c: Hypothalamus — liftover map ──────────────────────────────────────
print("  Building hypo → PFC liftover map…", flush=True)
with open(HYPO_CACHE / "hypo_pseudobulks.pkl", "rb") as f:
    hypo = pickle.load(f)
with open(HYPO_CACHE / "mouse_peak_liftover.pkl", "rb") as f:
    liftover = pickle.load(f)   # list: per hypo-peak → None | (chrom, pos)

# Encode mouse age letters: Y=1, M=2, A=3
age_map = {"Y": 1.0, "M": 2.0, "A": 3.0}
hypo_age = np.array([age_map.get(a, np.nan) for a in hypo["age_letters"]])

# Build sorted PFC peak intervals for fast overlap lookup
pfc_start_by_chrom = {}
pfc_end_by_chrom   = {}
pfc_idx_by_chrom   = {}
pfc_runmax_by_chrom = {}
for i, (c, s, e) in enumerate(zip(chroms, starts, ends)):
    pfc_start_by_chrom.setdefault(c, []).append((s, e, i))

# Sort each chromosome by start
for c in pfc_start_by_chrom:
    pfc_start_by_chrom[c].sort()
    arr = np.array(pfc_start_by_chrom[c])
    pfc_start_by_chrom[c] = arr[:, 0].astype(np.int32)
    pfc_end_by_chrom[c]   = arr[:, 1].astype(np.int32)
    pfc_idx_by_chrom[c]   = arr[:, 2].astype(np.int32)
    # Running max of end-position seen so far (index 0..i): used to early-exit
    # the backward scan in find_pfc_peak below once no earlier-starting peak
    # could possibly still overlap pos (perf_find_pfc_peak_bug: naive backward
    # scan is O(n) per query without this).
    pfc_runmax_by_chrom[c] = np.maximum.accumulate(pfc_end_by_chrom[c])

def find_pfc_peak(chrom, pos):
    """Return PFC peak index for a liftover point, or -1 if no overlap."""
    if chrom not in pfc_start_by_chrom:
        return -1
    ss = pfc_start_by_chrom[chrom]
    ee = pfc_end_by_chrom[chrom]
    ii = pfc_idx_by_chrom[chrom]
    rm = pfc_runmax_by_chrom[chrom]
    idx = np.searchsorted(ss, pos + 1, side="left") - 1
    if idx < 0:
        return -1
    while idx >= 0 and ss[idx] <= pos:
        if ee[idx] > pos:
            return int(ii[idx])
        if rm[idx] <= pos:
            break  # no peak at or before idx can overlap pos either
        idx -= 1
    return -1

# Map each hypo peak to a PFC peak
hypo2pfc = {}   # hypo_peak_idx → pfc_peak_idx
for hi, entry in enumerate(liftover):
    if entry is None:
        continue
    chrom, pos = entry
    pi = find_pfc_peak(chrom, int(pos))
    if pi >= 0:
        hypo2pfc.setdefault(pi, []).append(hi)

print(f"  Hypo peaks with PFC overlap: {len(hypo2pfc):,} PFC peaks mapped")

# Compute hypo CPM+log1p
X_hypo = cpm_log1p(hypo["all_cells"]["sum"])    # (20, 313013)
valid_hypo_age = np.isfinite(hypo_age)

# For each PFC peak with hypo match: average hypo peaks, then correlate
r_hypo_arr = np.full(n_peaks, np.nan, dtype=np.float32)
for pfc_idx, hypo_idxs in hypo2pfc.items():
    hypo_expr = X_hypo[:, hypo_idxs].mean(1)   # average if multiple
    m = valid_hypo_age
    if m.sum() < 3:
        continue
    r = pearson_r(hypo_age[m], hypo_expr[m])
    r_hypo_arr[pfc_idx] = r

del X_hypo; gc.collect()

# ── 3d: Combine, save ─────────────────────────────────────────────────────
print("  Combining and saving…", flush=True)
df_corr = pd.DataFrame({
    "peak_name": peak_names,
    "r_pfc":     r_pfc.astype(np.float32),
    "r_hip":     r_hip.astype(np.float32),
    "r_hypo":    r_hypo_arr,
    "gene":      [idx2gene.get(i, "") for i in range(n_peaks)],
})
# mean r across available datasets (require ≥2 non-NaN, prefer all 3)
vals = df_corr[["r_pfc","r_hip","r_hypo"]].values.astype(np.float64)
n_valid = np.sum(~np.isnan(vals), axis=1)
mean_r  = np.nanmean(vals, axis=1)
mean_r[n_valid < 2] = np.nan
df_corr["mean_r"] = mean_r.astype(np.float32)
df_corr["n_datasets"] = n_valid

df_corr.to_csv(RESULTS / "cross_dataset_corr.csv", index=False)
print(f"Saved {RESULTS/'cross_dataset_corr.csv'}  ({len(df_corr):,} peaks)")

# Top 5 peaks ranked by mean_r (all 3 datasets present, highest mean r)
top5 = (df_corr[df_corr.n_datasets == 3]
        .sort_values("mean_r", ascending=False)
        .head(5).reset_index(drop=True))
print("\nTop 5 cross-dataset peaks:")
print(top5[["peak_name","gene","r_pfc","r_hip","r_hypo","mean_r"]].to_string())

for _, r in top5.iterrows():
    log_test(
        "pfc_internal_valid", "cross_dataset_top_peaks", analysis="cross-dataset age correlation",
        test="pearson_correlation", group_a="PFC (N=357)", group_b=r["peak_name"],
        n_a=357, statistic=r["r_pfc"], notes=f"gene={r['gene']}",
    )
    log_test(
        "pfc_internal_valid", "cross_dataset_top_peaks", analysis="cross-dataset age correlation",
        test="pearson_correlation", group_a="Hippocampus (N=40)", group_b=r["peak_name"],
        n_a=40, statistic=r["r_hip"], notes=f"gene={r['gene']}",
    )
    log_test(
        "pfc_internal_valid", "cross_dataset_top_peaks", analysis="cross-dataset age correlation",
        test="pearson_correlation", group_a="Mouse hypothalamus (N=20)", group_b=r["peak_name"],
        n_a=20, statistic=r["r_hypo"], notes=f"gene={r['gene']}",
    )


# ══════════════════════════════════════════════════════════════════════════
# Part 3e: Cross-dataset scatter/violin plot
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Part 3e: Cross-dataset plots ===")

# Need raw expressions for top 5 peaks
top5_idx = [int(df_corr[df_corr.peak_name == pn].index[0]) for pn in top5.peak_name]

# Re-load CPM+log1p for PFC and hip (sliced to top5 peaks)
X_pfc_top = cpm_log1p(pfc_all["peak_sum"][:, top5_idx])
X_hip_top  = cpm_log1p(hip["all_cells"]["peak_sum"][:, top5_idx])

# Hypo: average matched hypo peaks for each top5 PFC peak
hypo_sum_raw = hypo["all_cells"]["sum"]      # (20, 313013)
X_hypo_raw   = cpm_log1p(hypo_sum_raw)       # (20, 313013)
X_hypo_top   = np.zeros((20, 5), dtype=np.float32)
for col, pi in enumerate(top5_idx):
    hypo_idxs = hypo2pfc.get(pi, [])
    if hypo_idxs:
        X_hypo_top[:, col] = X_hypo_raw[:, hypo_idxs].mean(1)

HYPO_AGE_ORDER = ["Y", "M", "A"]
HYPO_AGE_LABEL = {"Y": "Young", "M": "Middle", "A": "Aged"}

n_top = len(top5)
fig, axes = plt.subplots(n_top, 3, figsize=(11, n_top * 2.8))

for row_i, (_, peak_row) in enumerate(top5.iterrows()):
    pname = peak_row["peak_name"]
    gene  = peak_row["gene"] or ""
    col   = row_i  # column in top5_idx

    # ── PFC scatter ──
    ax = axes[row_i, 0]
    expr_pfc = X_pfc_top[:, col]
    r_p = float(peak_row["r_pfc"])
    ax.scatter(pfc_age, expr_pfc, s=12, alpha=0.5, color="#E64B35", edgecolors="none")
    m = np.isfinite(expr_pfc)
    z = np.polyfit(pfc_age[m], expr_pfc[m], 1)
    xl = np.array([pfc_age.min(), pfc_age.max()])
    ax.plot(xl, np.polyval(z, xl), "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Age (years)", fontsize=7)
    ax.set_ylabel("log-CPM", fontsize=7)
    ax.set_title(f"PFC (N=357)", fontsize=7.5, pad=2)
    ax.text(0.04, 0.96, f"r = {r_p:.3f}", transform=ax.transAxes,
            va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    ax.tick_params(labelsize=6)
    if row_i == 0:
        ax.set_title("PFC (N=357)", fontsize=8, pad=3)

    # Row label
    ax.set_ylabel(f"{pname}\n({gene})\nlog-CPM", fontsize=6.5)

    # ── Hippocampus scatter ──
    ax = axes[row_i, 1]
    expr_hip = X_hip_top[:, col]
    r_h = float(peak_row["r_hip"])
    ax.scatter(hip_age, expr_hip, s=16, alpha=0.65, color="#3C5488", edgecolors="none")
    m2 = np.isfinite(expr_hip)
    z2 = np.polyfit(hip_age[m2], expr_hip[m2], 1)
    xl2 = np.array([hip_age.min(), hip_age.max()])
    ax.plot(xl2, np.polyval(z2, xl2), "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Age (years)", fontsize=7)
    ax.set_ylabel("log-CPM", fontsize=7)
    ax.text(0.04, 0.96, f"r = {r_h:.3f}", transform=ax.transAxes,
            va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    ax.tick_params(labelsize=6)
    if row_i == 0:
        ax.set_title("Hippocampus (N=40)", fontsize=8, pad=3)

    # ── Hypothalamus violin ──
    ax = axes[row_i, 2]
    expr_hypo = X_hypo_top[:, col]
    age_letters_arr = np.array(hypo["age_letters"])
    r_hy = float(peak_row["r_hypo"]) if np.isfinite(peak_row["r_hypo"]) else np.nan

    groups    = [HYPO_AGE_LABEL[ag] for ag in HYPO_AGE_ORDER]
    group_dat = [expr_hypo[age_letters_arr == ag] for ag in HYPO_AGE_ORDER]
    vparts = ax.violinplot(group_dat, positions=range(len(groups)),
                           showmedians=True, showextrema=False)
    for i, (pc, ag) in enumerate(zip(vparts["bodies"], HYPO_AGE_ORDER)):
        pc.set_facecolor(["#4DBBD5","#F39B7F","#E64B35"][i])
        pc.set_alpha(0.7)
        # overlay jitter
        jitter = np.random.default_rng(42+i).uniform(-0.12, 0.12, len(group_dat[i]))
        ax.scatter(np.full(len(group_dat[i]), i) + jitter, group_dat[i],
                   s=12, alpha=0.8, color=["#4DBBD5","#F39B7F","#E64B35"][i],
                   edgecolors="none", zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7)
    ax.set_ylabel("log-CPM", fontsize=7)
    ax.tick_params(labelsize=6)
    rlab = f"r = {r_hy:.3f}" if np.isfinite(r_hy) else "r = n/a"
    ax.text(0.04, 0.96, rlab, transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    if row_i == 0:
        ax.set_title("Hypothalamus (N=20, mouse)", fontsize=8, pad=3)

fig.suptitle("Top cross-dataset age-correlated peaks (PFC, Hippocampus, Mouse Hypothalamus)",
             fontsize=10, y=1.01)
fig.tight_layout()
for ext in ("pdf","png"):
    p = FIG / f"cross_dataset_top_peaks.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150); print(f"Saved {p}")
plt.close(fig)

print("\nDone.")
