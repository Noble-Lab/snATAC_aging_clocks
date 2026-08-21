"""
Clock-specific SHAP top-N cross-cohort Spearman analysis — 4 experiments.

For each cross-cohort clock, uses the exact feature space and normalization
of that clock to train Ridge, compute SHAP = |coef| * std(X_train_z), then
selects top-N features and plots Spearman(CPM_log1p, age) in PFC vs target.

Experiments (all-cells only):
  PFC-HIP     : HIP 5kb HVP tiles overlapping PFC peaks (overlap matrix)
  PFC-Hypo    : PFC peaks overlapped by lifted hypo mm10 peaks (intersect_all)
  PFC-Retinal : mm10 50kb tiles (44232 features)
  PFC-ZF      : danRer11 50kb tiles (from hg38→danRer11 liftover, 6377 features)

Clock feature spaces mirrored:
  ~/pfc_to_hippocampus/pfc_to_hip_atac_clock.py
  ~/pfc_to_hypo/pfc_to_hypo_atac_clock.py
  ~/pfc_to_retinal/multiome_allcell_clock_per_ct.py
  ~/pfc_to_zebrafish_retinal/zebrafish_allcell_clock_per_ct.py

Output: 1 combined figure (4 rows × 3 cols) + 4 individual figures per experiment.
"""
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
from stats_log import log_test

import numpy as np
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
BUNDLE       = ROOT.parent  # code/ -- other projects live alongside this one
ATAC_ROOT    = Path("~/atac_processing_techniques/cache")  # 758MB+, external
HIP_CACHE    = Path("~/pfc_to_hippocampus/cache")          # 579MB+, external
HYPO_CACHE   = BUNDLE / "pfc_to_hypo" / "cache"                         # small, in-repo
MOUSE_CACHE  = Path("~/pfc_to_mouse/cache")          # sibling project, not in this bundle
RET_CACHE    = Path("~/pfc_to_retinal/cache")              # GB-scale, external
ZF_CACHE     = Path("~/pfc_to_zebrafish_retinal/cache")    # GB-scale, external
FIG          = ROOT / "figures"
FIG.mkdir(exist_ok=True)

RIDGE_ALPHAS = tuple(np.logspace(-2, 6, 30))
N_VALUES     = [10, 15, 20, 50]


# ── helpers ────────────────────────────────────────────────────────────────
def cpm_log1p(X):
    X = np.asarray(X, dtype=np.float32)
    d = X.sum(1, keepdims=True)
    d = np.where(d < 1, 1.0, d)
    return np.log1p(X / d * 1e6)


def zscore_rows(X):
    """CPM+log1p + per-row z-score (matches all 4 clock normalizations)."""
    X = cpm_log1p(X).astype(np.float64)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-8
    return (X - mu) / sd


def spearman_vec(X, y):
    """Vectorised Spearman rho: X (n×k), y (n,) → (k,)."""
    y_rank = rankdata(y, method="average").astype(np.float64)
    X_rank = np.apply_along_axis(
        lambda c: rankdata(c, method="average"), 0, X
    ).astype(np.float64)
    Xc = X_rank - X_rank.mean(0)
    yc = y_rank - y_rank.mean()
    num = Xc.T @ yc
    denom = np.sqrt((Xc**2).sum(0)) * (np.sqrt((yc**2).sum()) + 1e-12)
    return (num / (denom + 1e-12)).astype(np.float32)


def clock_shap(X_raw, y):
    """Train Ridge on z-scored X, return shap = |coef| * std(X_z, axis=0)."""
    X_z = zscore_rows(X_raw)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_z, y)
    return np.abs(ridge.coef_) * X_z.std(0)


def run_experiment(pfc_X_raw, pfc_age, test_X_raw, test_age):
    """
    Returns dict: N → {"pfc": rho_array, "test": rho_array}
    Uses CPM+log1p (no z-score) for Spearman computation.
    """
    shap   = clock_shap(pfc_X_raw, pfc_age)
    pfc_cpm  = cpm_log1p(pfc_X_raw)
    test_cpm = cpm_log1p(test_X_raw)
    results  = {}
    for N in N_VALUES:
        top_n = np.argsort(shap)[::-1][:N]
        results[N] = {
            "pfc":  spearman_vec(pfc_cpm[:, top_n], pfc_age),
            "test": spearman_vec(test_cpm[:, top_n], test_age),
            "shap_range": (float(shap[top_n[-1]]), float(shap[top_n[0]])),
        }
    return results


def plot_row(axes_row, results, y_label, row_title):
    """Plot one row (4 panels for N=10/15/20/50)."""
    for col_i, (ax, N) in enumerate(zip(axes_row, N_VALUES)):
        rp = results[N]["pfc"]
        rt = results[N]["test"]
        mask = np.isfinite(rp) & np.isfinite(rt)
        xm, ym = rp[mask], rt[mask]

        lim = (max(np.abs(xm).max(), np.abs(ym).max()) * 1.1
               if mask.sum() > 0 else 1.0)
        lim = min(lim, 1.0)

        # Quadrant backgrounds: green = signs agree, red = signs disagree
        ax.fill_between([0,    lim], 0,    lim,  color="#2ca02c", alpha=0.10, zorder=0)  # Q1
        ax.fill_between([-lim, 0],  -lim,  0,   color="#2ca02c", alpha=0.10, zorder=0)  # Q3
        ax.fill_between([-lim, 0],  0,    lim,  color="#d62728", alpha=0.10, zorder=0)  # Q2
        ax.fill_between([0,    lim], -lim,  0,   color="#d62728", alpha=0.10, zorder=0)  # Q4

        agree = (xm * ym) >= 0
        dot_colors = np.where(agree, "#2ca02c", "#d62728")
        ax.scatter(xm, ym, s=25, alpha=0.8, c=dot_colors,
                   edgecolors="none", rasterized=True)

        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.5", lw=0.6, zorder=1)
        ax.axvline(0, color="0.5", lw=0.6, zorder=1)

        if mask.sum() >= 3:
            rho_ov, pval = spearmanr(xm, ym)
        else:
            rho_ov, pval = np.nan, np.nan
        log_test(
            "pfc_internal_valid", "shap_clock_specific_spearman_combined",
            analysis=f"{row_title}: top-{N} SHAP peak Spearman rho concordance",
            test="spearman_correlation", group_a="PFC rho", group_b=row_title,
            n_a=int(mask.sum()), statistic=rho_ov, p_value=pval,
            notes=f"N_top_peaks={N}",
        )
        p_str = (f"p={pval:.2e}" if pval < 0.001
                 else f"p={pval:.3f}") if np.isfinite(pval) else ""
        n_agree = int(agree.sum())
        ax.text(0.04, 0.96,
                f"ρ={rho_ov:.3f}\n{p_str}\nagree: {n_agree}/{mask.sum()}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.8", alpha=0.85))
        ax.set_xlabel("PFC rho", fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        title = f"{row_title} — top {N}" if col_i == 0 else f"top {N}"
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.tick_params(labelsize=6)


# ══════════════════════════════════════════════════════════════════════════
# Load shared PFC pseudobulk (needed for HIP + Hypo projections)
# ══════════════════════════════════════════════════════════════════════════
print("Loading PFC pseudobulk…")
with open(ATAC_ROOT / "pfc_peak_pseudobulk.pkl", "rb") as f:
    pfc_raw = pickle.load(f)
pfc_sum  = pfc_raw["peak_sum"]          # (357, 521217) float32
pfc_age  = pfc_raw["age"].astype(np.float64)
pfc_cc   = pfc_raw["cell_counts"]       # (357,)
print(f"  PFC: {len(pfc_age)} donors, {pfc_sum.shape[1]:,} peaks")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: PFC-HIP
# Feature space: HIP 5kb HVP tiles overlapping PFC peaks
# ══════════════════════════════════════════════════════════════════════════
print("\n=== PFC-HIP ===")
print("  Loading HIP overlap matrix…")
with open(HIP_CACHE / "pfc_peaks_to_hip_tiles_M.pkl", "rb") as f:
    M_hip = pickle.load(f)   # sparse (521217, 5000)
col_sum   = np.asarray(M_hip.sum(0)).ravel()
keep_idx  = np.where(col_sum > 0)[0]    # 4991 intersecting tiles
M_keep    = M_hip[:, keep_idx]          # (521217, 4991) sparse
del M_hip; gc.collect()
print(f"  Intersecting tiles: {len(keep_idx):,}")

print("  Projecting PFC peaks → HIP tile space…")
# (4991, 521217) @ (521217, 357) → (4991, 357) → (357, 4991)
pfc_tile_hip = M_keep.T.dot(pfc_sum.T).T.astype(np.float32)  # (357, 4991)

print("  Loading HIP pseudobulks…")
with open(HIP_CACHE / "hip_pseudobulks.pkl", "rb") as f:
    hip = pickle.load(f)
hip_tile  = hip["all_cells"]["tile_sum"][:, keep_idx].astype(np.float32)  # (40, 4991)
hip_cc    = hip["all_cells"]["cell_counts"]   # (40,)
hip_age   = hip["age"].astype(np.float64)     # (40,)
valid_hip = hip_cc > 0
hip_tile_v = hip_tile[valid_hip]
hip_age_v  = hip_age[valid_hip]
del hip, hip_tile, M_keep; gc.collect()
print(f"  HIP: {valid_hip.sum()} valid donors, ages {hip_age_v.min():.0f}–{hip_age_v.max():.0f}")

res_hip = run_experiment(pfc_tile_hip, pfc_age, hip_tile_v, hip_age_v)
for N in N_VALUES:
    lo, hi_ = res_hip[N]["shap_range"]
    print(f"  N={N}: SHAP {lo:.4f}–{hi_:.4f}")
del pfc_tile_hip, hip_tile_v; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: PFC-Hypo
# Feature space: PFC peaks covered by lifted hypo mm10 peaks (intersect_all)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== PFC-Hypo ===")
print("  Loading hypo pseudobulks + liftover…")
with open(HYPO_CACHE / "hypo_pseudobulks.pkl", "rb") as f:
    hypo = pickle.load(f)
hypo_sum    = hypo["all_cells"]["sum"].astype(np.float32)   # (20, 313013)
hypo_cc     = hypo["all_cells"]["cc"]
hypo_age    = np.array([{"Y": 1.0, "M": 2.0, "A": 3.0}.get(a, np.nan)
                         for a in hypo["age_letters"]])
valid_hypo  = hypo_cc > 0
hypo_sum_v  = hypo_sum[valid_hypo]
hypo_age_v  = hypo_age[valid_hypo]
print(f"  Hypo: {valid_hypo.sum()} valid samples (Y/M/A ordinal age)")
del hypo; gc.collect()

# Reconstruct hypo→PFC mapping from liftover (same as intersect_all strategy)
print("  Building hypo→PFC liftover mapping…")
with open(HYPO_CACHE / "mouse_peak_liftover.pkl", "rb") as f:
    liftover = pickle.load(f)   # list len=313013, (chrom, pos) or None

peak_names = list(pfc_raw["peak_names"])
starts_by_ch, ends_by_ch, idx_by_ch = {}, {}, {}
for i, name in enumerate(peak_names):
    s = str(name)
    if ":" in s:
        c, rest = s.split(":", 1); a, b = rest.split("-")
    else:
        parts = s.rsplit("-", 2); c, a, b = parts[0], parts[1], parts[2]
    c, a, b = c, int(a), int(b)
    starts_by_ch.setdefault(c, []).append((a, b, i))
for c in starts_by_ch:
    arr = np.array(sorted(starts_by_ch[c]))
    starts_by_ch[c] = arr[:, 0].astype(np.int32)
    ends_by_ch[c]   = arr[:, 1].astype(np.int32)
    idx_by_ch[c]    = arr[:, 2].astype(np.int32)

# NOTE: originally a per-liftover-entry find_pfc_peak() call with a Python
# `while` backward scan -- on a miss this degenerates to an O(n_peaks_on_
# chrom) walk all the way to index 0 (documented perf bug; confirmed via
# py-spy to stall for 15-20+ minutes here under concurrent machine load).
# PFC peaks are verified non-overlapping (0/521217 adjacent-pair overlaps
# within any chromosome; see scripts/shap_enrichment_crossdataset.py's
# cross-dataset-corr step, where this same optimization was validated to
# give bit-identical output to the slow reference implementation), so at
# most one peak can ever contain a given position -- checking only the
# single searchsorted candidate is equivalent to the full backward scan,
# just O(1) instead of O(n) per query and fully vectorized per chromosome.
mouse_chrom_arr = np.array([e[0] if e is not None else "" for e in liftover])
mouse_pos_arr = np.array([e[1] if e is not None else -1 for e in liftover], dtype=np.int64)
valid_lift = mouse_chrom_arr != ""

hypo2pfc = {}    # pfc_peak_idx → [hypo_peak_idx, ...]
for c in np.unique(mouse_chrom_arr[valid_lift]):
    if c not in starts_by_ch:
        continue
    ss, ee, ii = starts_by_ch[c], ends_by_ch[c], idx_by_ch[c]
    q_mask = valid_lift & (mouse_chrom_arr == c)
    q_hidx = np.nonzero(q_mask)[0]
    q_pos = mouse_pos_arr[q_hidx]
    insert = np.searchsorted(ss, q_pos + 1, side="left") - 1
    ok = insert >= 0
    contains = np.zeros(len(q_pos), dtype=bool)
    contains[ok] = ee[insert[ok]] > q_pos[ok]
    for hi, pi in zip(q_hidx[contains], ii[insert[contains]]):
        hypo2pfc.setdefault(int(pi), []).append(int(hi))
del liftover

pfc_valid_hypo = sorted(hypo2pfc.keys())
n_hypo_feat    = len(pfc_valid_hypo)
print(f"  PFC peaks with hypo coverage: {n_hypo_feat:,}")

# Build sparse mapping M_m: (313013, n_hypo_feat) — maps hypo peaks → PFC feat cols
rows_m, cols_m = [], []
for j, pi in enumerate(pfc_valid_hypo):
    for hi in hypo2pfc[pi]:
        rows_m.append(hi); cols_m.append(j)
M_m = sp.csr_matrix(
    (np.ones(len(rows_m), np.float32), (rows_m, cols_m)),
    shape=(hypo_sum_v.shape[1], n_hypo_feat)
)
del rows_m, cols_m, hypo2pfc

# PFC features: subset of PFC peaks in the intersecting set
pfc_feat_hypo = pfc_sum[:, pfc_valid_hypo].astype(np.float32)  # (357, n_hypo_feat)

# Hypo features: project via M_m (sum of hypo peaks → each PFC peak column)
print("  Projecting hypo peaks → PFC peak space…")
hypo_feat = (hypo_sum_v @ M_m).astype(np.float32)  # (valid, n_hypo_feat)
del hypo_sum, hypo_sum_v, M_m; gc.collect()
print(f"  Hypo feat matrix: {hypo_feat.shape}")

res_hypo = run_experiment(pfc_feat_hypo, pfc_age, hypo_feat, hypo_age_v)
for N in N_VALUES:
    lo, hi_ = res_hypo[N]["shap_range"]
    print(f"  N={N}: SHAP {lo:.4f}–{hi_:.4f}")
del pfc_feat_hypo, hypo_feat; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# Free PFC peak sum — no longer needed (retinal/ZF use tile-space caches)
# ══════════════════════════════════════════════════════════════════════════
del pfc_sum, pfc_raw; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: PFC-Retinal
# Feature space: mm10 50kb tiles (pfc_to_mouse cache, 44232 features)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== PFC-Retinal ===")
print("  Loading PFC mm10 tile matrix (44232 tiles)…")
with open(MOUSE_CACHE / "pfc_X_tile50kb.pkl", "rb") as f:
    pfc_X_ret = np.asarray(pickle.load(f), dtype=np.float32)  # (357, 44232)
with open(MOUSE_CACHE / "feat_idx_tile50kb.pkl", "rb") as f:
    feat_idx_ret = pickle.load(f)   # (44232,) global mm10 tile indices

print("  Loading retinal tile matrix…")
with open(RET_CACHE / "retinal_tile50kb.pkl", "rb") as f:
    ret = pickle.load(f)
ret_X   = ret["X_full"][:, feat_idx_ret].astype(np.float32)  # (17, 44232)
ret_age = ret["ages_wk"].astype(np.float64)
del ret; gc.collect()
print(f"  Retinal: {ret_X.shape[0]} samples, ages {ret_age.min():.0f}–{ret_age.max():.0f} wk")

res_ret = run_experiment(pfc_X_ret, pfc_age, ret_X, ret_age)
for N in N_VALUES:
    lo, hi_ = res_ret[N]["shap_range"]
    print(f"  N={N}: SHAP {lo:.4f}–{hi_:.4f}")
del pfc_X_ret, ret_X; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: PFC-Zebrafish
# Feature space: danRer11 50kb tiles (6377 unique tiles from hg38→danRer11 liftover)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== PFC-Zebrafish ===")
print("  Loading PFC danRer11 tile matrix (6377 tiles)…")
with open(ZF_CACHE / "pfc_X_danrer11_tile50kb.pkl", "rb") as f:
    pfc_X_zf = np.asarray(pickle.load(f), dtype=np.float32)  # (357, 6377)

print("  Loading zebrafish tile matrix…")
with open(ZF_CACHE / "zebrafish_tile50kb.pkl", "rb") as f:
    zf = pickle.load(f)
with open(ZF_CACHE / "pfc_to_danrer11_feat_idx.pkl", "rb") as f:
    zf_lift = pickle.load(f)
unique_tiles_zf = zf_lift["unique_tiles"]   # (6377,) global danRer11 tile indices
zf_X    = zf["X_full"][:, unique_tiles_zf].astype(np.float32)  # (13, 6377)
zf_age  = zf["ages_mo"].astype(np.float64)
del zf, zf_lift; gc.collect()
print(f"  Zebrafish: {zf_X.shape[0]} samples, ages {zf_age.min():.0f}–{zf_age.max():.0f} mo")

res_zf = run_experiment(pfc_X_zf, pfc_age, zf_X, zf_age)
for N in N_VALUES:
    lo, hi_ = res_zf[N]["shap_range"]
    print(f"  N={N}: SHAP {lo:.4f}–{hi_:.4f}")
del pfc_X_zf, zf_X; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    ("PFC-HIP",       res_hip,  "HIP rho (5kb HVP tiles)"),
    ("PFC-Hypo",      res_hypo, "Mouse Hypo rho"),
    ("PFC-Retinal",   res_ret,  "Mouse Retinal rho (mm10 50kb tiles)"),
    ("PFC-Zebrafish", res_zf,   "Zebrafish rho (danRer11 50kb tiles)"),
]

print("\nPlotting combined figure (4 rows × 4 cols)…")
fig, axes = plt.subplots(4, 4, figsize=(14.0, 14.0), squeeze=False)
for row_i, (name, results, y_label) in enumerate(EXPERIMENTS):
    plot_row(axes[row_i], results, y_label, row_title=name)

fig.suptitle(
    "Clock-specific SHAP top-N: PFC vs target cohort age correlations\n"
    "Each dot = one feature; axes = Spearman(CPM_log1p, age)\n"
    "SHAP from clock trained on that cohort's feature space",
    fontsize=10, y=1.01
)
fig.tight_layout(h_pad=3.0, w_pad=2.0)
for ext in ("pdf", "png"):
    p = FIG / f"shap_clock_specific_spearman_combined.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"  Saved {p.name}")
plt.close(fig)

print("Plotting individual figures…")
for name, results, y_label in EXPERIMENTS:
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), squeeze=False)
    plot_row(axes[0], results, y_label, row_title=name)
    fig.suptitle(
        f"{name} — clock-specific SHAP peaks, Spearman cross-cohort",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    tag = name.lower().replace("-", "_")
    for ext in ("pdf", "png"):
        p = FIG / f"shap_clock_specific_{tag}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")

print("\nDone.")
