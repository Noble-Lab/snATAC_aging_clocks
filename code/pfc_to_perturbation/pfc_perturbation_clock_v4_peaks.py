#!~/micromamba/envs/sat/bin/python
"""
PFC ATAC Clock → GSE294103 SIRT6 OE — peak-level features
===========================================================
Lifts SIRT6 mm10 peaks to hg38 via pyliftover,
intersects with PFC 500bp peaks, and trains/predicts at peak resolution.

Intersection logic:
  For each SIRT6 peak, lift its midpoint mm10→hg38.
  Check overlap with PFC 500bp peaks on the same chromosome.
  A peak pair overlaps if the lifted midpoint falls within the PFC window.

External large-file dependency (kept at a fixed absolute path, not inside
this repo): ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl
(238MB). The mm10->hg38 liftover chain is read from pfc_to_hypo's own cache/
(code/pfc_to_hypo/cache/mm10ToHg38.over.chain.gz), which is in-repo.
Everything else (cache_v4/, figures_v4/, results_v4/, logs/, data/) is local
to this script's own directory.
"""

import logging
import pickle
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test, log_coef

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyliftover import LiftOver
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeCV

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
FIG     = ROOT / "figures_v4"
RESULTS = ROOT / "results_v4"
LOG_DIR = ROOT / "logs"
CACHE   = ROOT / "cache_v4"

ATAC_META_PKL = Path("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl")
CHAIN_GZ      = ROOT.parent / "pfc_to_hypo" / "cache" / "mm10ToHg38.over.chain.gz"
DATA_CSV      = ROOT / "data" / "GSE294103_atac_seq_normalized_reads.csv.gz"

RIDGE_ALPHAS = tuple(np.logspace(-2, 8, 40))

for d in (FIG, RESULTS, LOG_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "perturbation_v4_peaks.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger()

# ── Group metadata ─────────────────────────────────────────────────────────────
GROUP_ORDER  = ["Young WT", "Young SIRT6", "Old WT", "Old SIRT6"]
GROUP_COLORS = {
    "Young WT":    "#4393C3",
    "Young SIRT6": "#92C5DE",
    "Old WT":      "#D6604D",
    "Old SIRT6":   "#762A83",
}


def _sample_to_group(col: str) -> str:
    if col.startswith("young_wt"):    return "Young WT"
    if col.startswith("young_sirt6"): return "Young SIRT6"
    if col.startswith("old_wt"):      return "Old WT"
    if col.startswith("old_sirt6"):   return "Old SIRT6"
    return "Unknown"


# ── Normalization ──────────────────────────────────────────────────────────────
def normalize(X: np.ndarray) -> np.ndarray:
    X  = np.asarray(X, dtype=np.float64)
    d  = X.sum(1, keepdims=True).clip(1e-9)
    X  = np.log1p(X / d * 1e6)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


# ── Load PFC peaks ─────────────────────────────────────────────────────────────
def load_pfc_peaks():
    log.info("Loading PFC peak data …")
    with open(ATAC_META_PKL, "rb") as f:
        meta = pickle.load(f)
    peak_names = meta["peak_names"]   # list of "chrN:start-end" (hg38, 500bp windows)
    peak_sum   = meta["peak_sum"]     # (357, 521217) float32
    ages       = np.array(meta["age"])

    # parse peak coords into arrays per chromosome
    log.info(f"  PFC: {len(peak_names):,} peaks, {len(ages)} donors")
    return ages, peak_sum, peak_names


# ── Parse PFC peak names → per-chr arrays ─────────────────────────────────────
def build_pfc_index(peak_names: list) -> dict:
    """Returns dict chr → (starts_array, ends_array, global_idx_array)."""
    chr_data = {}
    for gi, name in enumerate(peak_names):
        ch, coords = name.split(":")
        s, e = coords.split("-")
        if ch not in chr_data:
            chr_data[ch] = ([], [], [])
        chr_data[ch][0].append(int(s))
        chr_data[ch][1].append(int(e))
        chr_data[ch][2].append(gi)
    # convert to numpy for fast searchsorted
    result = {}
    for ch, (ss, ee, gi) in chr_data.items():
        ss = np.array(ss, dtype=np.int32)
        ee = np.array(ee, dtype=np.int32)
        gi = np.array(gi, dtype=np.int32)
        order = np.argsort(ss)
        result[ch] = (ss[order], ee[order], gi[order])
    return result


# ── Read SIRT6 CSV ─────────────────────────────────────────────────────────────
def read_sirt6_csv():
    log.info(f"Reading {DATA_CSV.name} …")
    import gzip
    with gzip.open(DATA_CSV, "rb") as raw:
        content = raw.read()
    if content[0] == 0xFF:
        content = content[1:]
    text = content.decode("utf-8")
    df = pd.read_csv(pd.io.common.StringIO(text))
    df.rename(columns={df.columns[0]: "Chr"}, inplace=True)
    sample_cols = [c for c in df.columns if c.startswith(("young_", "old_"))]
    log.info(f"  Peaks: {len(df):,}  Samples: {len(sample_cols)}")
    return df, sample_cols


# ── Liftover + intersect → build feature matrices ─────────────────────────────
def build_peak_matrices(df_sirt6, sample_cols, pfc_index):
    cache_pkl = CACHE / "gse294103_peak_matrices.pkl"
    if cache_pkl.exists():
        log.info("Loading cached peak matrices …")
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    log.info("Loading liftover chain (mm10→hg38) …")
    lo = LiftOver(str(CHAIN_GZ))

    n_sirt6_peaks = len(df_sirt6)
    n_samples     = len(sample_cols)

    # For each SIRT6 peak we need: (lifted_chr, lifted_midpoint)
    # Then find overlapping PFC peak indices
    matched_pfc_idx   = []   # global PFC peak index
    matched_sirt6_idx = []   # row index in df_sirt6

    log.info("Lifting SIRT6 peaks mm10→hg38 and intersecting with PFC peaks …")
    skipped_liftover = 0
    skipped_no_match = 0

    for row_i, row in enumerate(df_sirt6.itertuples(index=False)):
        raw_chr = str(row.Chr).strip()
        if raw_chr in ("X", "Y"):
            mm10_ch = "chr" + raw_chr
        else:
            try:
                mm10_ch = "chr" + str(int(float(raw_chr)))
            except (ValueError, TypeError):
                skipped_liftover += 1
                continue

        start = int(row.start_peak)
        end   = int(row.end_peak)
        mid   = (start + end) // 2

        result = lo.convert_coordinate(mm10_ch, mid)
        if not result:
            skipped_liftover += 1
            continue

        hg38_ch, hg38_pos, *_ = result[0]

        if hg38_ch not in pfc_index:
            skipped_no_match += 1
            continue

        ss, ee, gi = pfc_index[hg38_ch]
        # find PFC peaks whose start <= hg38_pos < end
        idx_right = np.searchsorted(ss, hg38_pos, side="right")
        # check peaks to the left of idx_right whose end > hg38_pos
        # (ss is sorted, so go back from idx_right)
        for k in range(max(0, idx_right - 1), -1, -1):
            if ss[k] > hg38_pos:
                continue
            if ee[k] > hg38_pos:
                matched_pfc_idx.append(gi[k])
                matched_sirt6_idx.append(row_i)
                break   # each SIRT6 midpoint overlaps at most one PFC peak
            break       # ss is sorted; earlier peaks won't overlap either

    log.info(f"  Skipped (liftover fail): {skipped_liftover:,}")
    log.info(f"  Skipped (no PFC overlap): {skipped_no_match:,}")
    log.info(f"  Matched peak pairs: {len(matched_pfc_idx):,}")

    # unique PFC peak indices (some may be hit by >1 SIRT6 peak; sum them)
    matched_pfc_idx   = np.array(matched_pfc_idx,   dtype=np.int32)
    matched_sirt6_idx = np.array(matched_sirt6_idx, dtype=np.int32)

    unique_pfc = np.unique(matched_pfc_idx)
    pfc_local  = {g: l for l, g in enumerate(unique_pfc)}
    n_feats    = len(unique_pfc)
    log.info(f"  Unique PFC features used: {n_feats:,}")

    # Build SIRT6 feature matrix: (n_samples, n_feats)
    sirt6_vals = df_sirt6[sample_cols].values.astype(np.float32)  # (n_peaks, n_samples)
    X_sirt6 = np.zeros((n_samples, n_feats), dtype=np.float32)
    for si, pi in zip(matched_sirt6_idx, matched_pfc_idx):
        li = pfc_local[pi]
        X_sirt6[:, li] += sirt6_vals[si, :]    # sum if multiple SIRT6 peaks → same PFC peak

    result = (unique_pfc, X_sirt6, sample_cols)
    with open(cache_pkl, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info("Cached peak matrices.")
    return result


# ── Train PFC clock on intersecting peaks ─────────────────────────────────────
def train_on_peaks(pfc_ages, pfc_peak_sum, unique_pfc):
    log.info("Training PFC Ridge clock on intersecting peaks …")
    X_pfc = pfc_peak_sum[:, unique_pfc].astype(np.float64)
    X_pfc_norm = normalize(X_pfc)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge.fit(X_pfc_norm, pfc_ages)
    log.info(f"  Ridge alpha={ridge.alpha_:.3g}  features={X_pfc.shape[1]:,}")
    for rank, i in enumerate(np.argsort(np.abs(ridge.coef_))[::-1][:200]):
        log_coef("pfc_to_perturbation", clock_name="gse294103_sirt6_peaks_ridge_clock",
                 feature=f"pfc_peak_{unique_pfc[i]}", coefficient=float(ridge.coef_[i]),
                 modality="ATAC", cell_type="All cells", rank=rank, model_type="RidgeCV")
    return ridge


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(ridge, X_sirt6, sample_cols) -> pd.DataFrame:
    X_norm = normalize(X_sirt6.astype(np.float64))
    preds  = ridge.predict(X_norm)
    rows   = [{"sample": s, "group": _sample_to_group(s), "pred_age": p}
              for s, p in zip(sample_cols, preds)]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gse294103_predictions_peaks.csv", index=False)
    log.info(f"Saved {len(df)} predictions.")
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]["pred_age"]
        log.info(f"  {g:15s}: mean={sub.mean():.2f}  sd={sub.std():.2f}  n={len(sub)}")
    return df


# ── Violin plot ────────────────────────────────────────────────────────────────
def plot_violin(df: pd.DataFrame, n_feats: int):
    fig, ax = plt.subplots(figsize=(6, 5))

    data_by_group = [df[df["group"] == g]["pred_age"].values for g in GROUP_ORDER]

    parts = ax.violinplot(
        data_by_group,
        positions=range(len(GROUP_ORDER)),
        showmedians=False, showextrema=False,
    )
    for i, (body, g) in enumerate(zip(parts["bodies"], GROUP_ORDER)):
        col = GROUP_COLORS[g]
        body.set_facecolor(col)
        body.set_edgecolor("0.3")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

        vals = data_by_group[i]
        jx   = np.random.uniform(i - 0.12, i + 0.12, len(vals))
        ax.scatter(jx, vals, color=col, s=40, edgecolors="0.25",
                   linewidths=0.5, zorder=4, alpha=0.85)

        med = np.median(vals)
        ax.hlines(med, i - 0.22, i + 0.22, colors=col, linewidths=2.5, zorder=5)
        ax.text(i + 0.26, med, f"{med:.1f}", fontsize=7.5,
                va="center", color=col, fontweight="bold")

    pairs = [
        ("Young WT", "Young SIRT6", 0, 1),
        ("Old WT",   "Old SIRT6",   2, 3),
        ("Young WT", "Old WT",      0, 2),
    ]
    y_max = df["pred_age"].max()
    y_min = df["pred_age"].min()
    step  = (y_max - y_min) * 0.08

    bracket_y = y_max + step
    for label_a, label_b, xi, xj in pairs:
        a = df[df["group"] == label_a]["pred_age"].values
        b = df[df["group"] == label_b]["pred_age"].values
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        log_test("pfc_to_perturbation", "gse294103_sirt6_violin_peaks",
                 analysis="PFC->GSE294103 SIRT6 peak Ridge clock, predicted age",
                 test="mannwhitneyu", group_a=label_a, group_b=label_b,
                 n_a=len(a), n_b=len(b), p_value=p)
        ax.plot([xi, xi, xj, xj],
                [bracket_y, bracket_y + step * 0.3, bracket_y + step * 0.3, bracket_y],
                lw=1.0, color="0.35")
        ax.text((xi + xj) / 2, bracket_y + step * 0.35,
                f"p={p:.3f} {sig}", ha="center", va="bottom",
                fontsize=7.0, color="0.25")
        bracket_y += step * 1.3

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, fontsize=9.5)
    ax.set_ylabel("Predicted age (human yr)", fontsize=10)
    ax.set_title(
        "GSE294103: SIRT6 overexpression — ATAC-seq\n"
        f"PFC-trained clock (peak-level, {n_feats:,} peaks after mm10→hg38 liftover)\n"
        "Horizontal bar = median",
        fontsize=9, pad=6,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG / f"gse294103_sirt6_violin_peaks.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        log.info(f"Saved {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pfc_ages, pfc_peak_sum, peak_names = load_pfc_peaks()
    pfc_index = build_pfc_index(peak_names)

    df_sirt6, sample_cols = read_sirt6_csv()

    unique_pfc, X_sirt6, sample_cols = build_peak_matrices(
        df_sirt6, sample_cols, pfc_index
    )

    ridge = train_on_peaks(pfc_ages, pfc_peak_sum, unique_pfc)
    df    = predict(ridge, X_sirt6, sample_cols)
    plot_violin(df, n_feats=len(unique_pfc))

    log.info("Done.")


if __name__ == "__main__":
    main()
