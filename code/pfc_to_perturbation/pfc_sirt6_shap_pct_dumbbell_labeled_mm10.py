#!~/micromamba/envs/sat/bin/python
"""
Dumbbell plot of the top 20 PFC->GSE294103 SIRT6 peak-level clock features
ranked by (SIRT6 - WT) mean|SHAP| percentile-rank difference, labeled with
their original mm10 SIRT6-peak coordinates.

pfc_perturbation_clock_v4_peaks.py's cache only stores the final
(unique_pfc, X_sirt6, sample_cols) tuple, not the intermediate mm10<->hg38
peak mapping, so this script rebuilds that mapping (liftover + intersect) to
recover, for each hg38 PFC feature used, the original mm10 SIRT6 peak(s)
that mapped onto it.

External large-file dependency (kept at a fixed absolute path, not inside
this repo): ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
(238MB). The mm10->hg38 liftover chain is read from pfc_to_hypo's own cache/
(code/pfc_to_hypo/cache/mm10ToHg38.over.chain.gz), which is in-repo.
Everything else (cache_v4/, figures_v4/, logs/, reference/) is local to this
script's own directory.
"""

import gzip
import io
import logging
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyliftover import LiftOver
from scipy.stats import rankdata

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
np.random.seed(42)

ROOT    = Path(__file__).resolve().parent
FIG     = ROOT / "figures_v4"
LOG_DIR = ROOT / "logs"
CACHE   = ROOT / "cache_v4"

ATAC_META_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
CHAIN_GZ      = ROOT.parent / "pfc_to_hypo" / "cache" / "mm10ToHg38.over.chain.gz"
TSS_FILE      = ROOT / "reference" / "hg38_gene_tss.tsv"  # bundled copy (892KB)
DATA_CSV      = ROOT / "data" / "GSE294103_atac_seq_normalized_reads.csv.gz"
PEAK_MAT_PKL  = CACHE / "gse294103_peak_matrices.pkl"
SHAP_PKL      = CACHE / "gse294103_shap_values.pkl"
MM10_MAP_PKL  = CACHE / "gse294103_hg38_to_mm10_map.pkl"

N_TOP = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sirt6_shap_pct_dumbbell_labeled_mm10.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()


def sample_group(col):
    return "SIRT6" if "sirt6" in col else "WT"


def build_tss_index(tss_df):
    idx = {}
    for ch, grp in tss_df.groupby("chr"):
        grp = grp.sort_values("tss")
        idx[ch] = (grp["tss"].values, grp["gene_name"].values)
    return idx


def nearest_gene(ch, mid, tss_idx):
    if ch not in tss_idx:
        return "unknown"
    tss_arr, name_arr = tss_idx[ch]
    i = np.searchsorted(tss_arr, mid)
    cands = []
    if i < len(tss_arr): cands.append((abs(tss_arr[i]   - mid), name_arr[i]))
    if i > 0:            cands.append((abs(tss_arr[i-1] - mid), name_arr[i-1]))
    return min(cands, key=lambda x: x[0])[1]


def nearest_genes_for_hg38_names(hg38_names, tss_idx):
    genes = []
    for name in hg38_names:
        ch, coords = name.split(":")
        s, e = coords.split("-")
        mid  = (int(s) + int(e)) // 2
        genes.append(nearest_gene(ch, mid, tss_idx))
    return genes


def build_pfc_index(peak_names):
    chr_data = {}
    for gi, name in enumerate(peak_names):
        ch, coords = name.split(":")
        s, e = coords.split("-")
        chr_data.setdefault(ch, ([], [], []))
        chr_data[ch][0].append(int(s))
        chr_data[ch][1].append(int(e))
        chr_data[ch][2].append(gi)
    result = {}
    for ch, (ss, ee, gi) in chr_data.items():
        ss = np.array(ss, dtype=np.int32)
        ee = np.array(ee, dtype=np.int32)
        gi = np.array(gi, dtype=np.int32)
        order = np.argsort(ss)
        result[ch] = (ss[order], ee[order], gi[order])
    return result


def read_sirt6_csv():
    log.info(f"Reading {DATA_CSV.name} …")
    with gzip.open(DATA_CSV, "rb") as raw:
        content = raw.read()
    if content[0] == 0xFF:
        content = content[1:]
    text = content.decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    df.rename(columns={df.columns[0]: "Chr"}, inplace=True)
    return df


def build_hg38_to_mm10_map(peak_names):
    """For every hg38 PFC peak index, recover the mm10 SIRT6 peak(s) that
    lifted onto it (rebuilds the same liftover+intersect logic used to
    build gse294103_peak_matrices.pkl, since only the final matrix — not
    this mapping — was cached)."""
    if MM10_MAP_PKL.exists():
        log.info("Loading cached hg38→mm10 map …")
        with open(MM10_MAP_PKL, "rb") as f:
            return pickle.load(f)

    log.info("Rebuilding hg38→mm10 peak map (liftover + intersect) …")
    pfc_index = build_pfc_index(peak_names)
    df_sirt6  = read_sirt6_csv()

    log.info("Loading liftover chain (mm10→hg38) …")
    lo = LiftOver(str(CHAIN_GZ))

    hg38_to_mm10 = {}   # global hg38 PFC peak idx -> list of "chrN:start-end" (mm10)

    for row in df_sirt6.itertuples(index=False):
        raw_chr = str(row.Chr).strip()
        if raw_chr in ("X", "Y"):
            mm10_ch = "chr" + raw_chr
        else:
            try:
                mm10_ch = "chr" + str(int(float(raw_chr)))
            except (ValueError, TypeError):
                continue

        start = int(row.start_peak)
        end   = int(row.end_peak)
        mid   = (start + end) // 2

        result = lo.convert_coordinate(mm10_ch, mid)
        if not result:
            continue
        hg38_ch, hg38_pos, *_ = result[0]
        if hg38_ch not in pfc_index:
            continue

        ss, ee, gi = pfc_index[hg38_ch]
        idx_right = np.searchsorted(ss, hg38_pos, side="right")
        for k in range(max(0, idx_right - 1), -1, -1):
            if ss[k] > hg38_pos:
                continue
            if ee[k] > hg38_pos:
                g = int(gi[k])
                hg38_to_mm10.setdefault(g, []).append(f"{mm10_ch}:{start}-{end}")
            break

    log.info(f"  Mapped {len(hg38_to_mm10):,} hg38 PFC peaks back to mm10 peak(s).")
    with open(MM10_MAP_PKL, "wb") as f:
        pickle.dump(hg38_to_mm10, f, protocol=4)
    return hg38_to_mm10


def compute_stats(shap_vals, sample_cols):
    abs_shap   = np.abs(shap_vals)
    wt_mask    = np.array([sample_group(s) == "WT"    for s in sample_cols])
    sirt6_mask = np.array([sample_group(s) == "SIRT6" for s in sample_cols])

    wt_mean    = abs_shap[wt_mask].mean(0)
    sirt6_mean = abs_shap[sirt6_mask].mean(0)

    wt_pct    = rankdata(wt_mean,    method="average") / len(wt_mean)    * 100
    sirt6_pct = rankdata(sirt6_mean, method="average") / len(sirt6_mean) * 100

    pct_diff  = sirt6_pct - wt_pct
    return wt_pct, sirt6_pct, pct_diff


def plot_diff_top20_pct_dumbbell_mm10(wt_pct, sirt6_pct, pct_diff, mm10_labels):
    top_idx   = np.argsort(pct_diff)[-N_TOP:][::-1]
    wt_p_vals = wt_pct[top_idx]
    s6_p_vals = sirt6_pct[top_idx]
    d_vals    = pct_diff[top_idx]
    labels    = [mm10_labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(N_TOP)

    for yi, (wt_v, s6_v) in enumerate(zip(wt_p_vals, s6_p_vals)):
        ax.plot([wt_v, s6_v], [yi, yi], color="0.72", linewidth=1.4, zorder=1)

    ax.scatter(wt_p_vals,  y, color="#F4A582", s=70, zorder=2, label="WT",
               edgecolors="0.35", linewidths=0.6)
    ax.scatter(s6_p_vals,  y, color="#92C5DE", s=70, zorder=2, label="SIRT6",
               edgecolors="0.35", linewidths=0.6)

    for yi, (wt_v, s6_v, dv) in enumerate(zip(wt_p_vals, s6_p_vals, d_vals)):
        x_mid = (wt_v + s6_v) / 2
        ax.text(x_mid, yi - 0.30, f"Δ={dv:+.1f}", ha="center", va="bottom",
                fontsize=7, color="0.15",
                fontweight="bold" if dv > 0 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.margins(y=0.03)
    ax.set_xlabel("Within-group percentile rank (%ile)", fontsize=9)
    ax.axvline(50, color="0.6", linewidth=0.7, linestyle="--", alpha=0.6, label="50th %ile")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(
        "Top 20 peaks by SIRT6 − WT mean |SHAP| percentile-rank difference\n"
        "(Δ labels = SIRT6 %ile − WT %ile; coordinates in original mm10)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "sirt6_shap_diff_top20_pct_dumbbell_labeled_mm10")


def _save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        log.info(f"Saved {p}")
    plt.close(fig)


def main():
    log.info("Loading cached peak matrices …")
    with open(PEAK_MAT_PKL, "rb") as f:
        unique_pfc, _X_sirt6, sample_cols = pickle.load(f)

    log.info("Loading peak names …")
    with open(ATAC_META_PKL, "rb") as f:
        meta = pickle.load(f)
    peak_names = meta["peak_names"]

    hg38_to_mm10 = build_hg38_to_mm10_map(peak_names)

    log.info("Loading TSS annotation …")
    tss_idx = build_tss_index(pd.read_csv(TSS_FILE, sep="\t"))

    log.info("Annotating peaks with nearest gene (hg38 locus) …")
    peak_names_used = [peak_names[i] for i in unique_pfc]
    gene_names = nearest_genes_for_hg38_names(peak_names_used, tss_idx)

    log.info("Loading cached SHAP values …")
    with open(SHAP_PKL, "rb") as f:
        shap_vals = pickle.load(f)

    wt_pct, sirt6_pct, pct_diff = compute_stats(shap_vals, sample_cols)

    # unique_pfc[i] is the global hg38 PFC peak index for local feature i
    mm10_labels = []
    for local_i, global_i in enumerate(unique_pfc):
        coords = hg38_to_mm10.get(int(global_i), ["unmapped"])
        mm10_labels.append(f"{gene_names[local_i]} ({' / '.join(coords)})")

    plot_diff_top20_pct_dumbbell_mm10(wt_pct, sirt6_pct, pct_diff, mm10_labels)

    log.info("Done.")


if __name__ == "__main__":
    main()
