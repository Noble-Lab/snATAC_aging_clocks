"""One-figure summary: age vs clock-contribution for the top IUT+posslope peak
per cell type (and all cells).

Selection: best pval_iut among peaks with min_gap > 0 AND positive HF slope.
Layout: 1 row × 7 panels (all_cells, Exc, Inh, Oligo, Astro, Mic, OPC).

Output:
  figures/dlpfc_top_iut_posslope_scatter_combined.pdf
"""
import gc
import pickle
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT        = Path(__file__).resolve().parent.parent
CACHE_DIR   = Path("~/atac_processing_techniques/cache")
GA_CACHE    = Path("~/age_accel_per_cell_type/cache")
RES         = ROOT / "results"
FIG         = ROOT / "figures"
SEAAD_META  = GA_CACHE  / "seaad_pseudobulks_by_ct.pkl"
SEA_AC_PKL  = CACHE_DIR / "seaad_peak_pseudobulk.pkl"
SEA_CT_PKL  = CACHE_DIR / "seaad_peak_pseudobulk_by_ct.pkl"
OVERLAP_NPZ = ROOT / "cache" / "pfc521k_x_seaad218k_overlap.npz"
PCT_CACHE   = ROOT / "cache" / "shap_sex_pct_cache.pkl"
COEF_CACHE  = ROOT / "cache" / "ridge_coef_cache.pkl"

CT_KEY = {
    "Exc": "Excitatory", "Inh": "Inhibitory", "Oligo": "Oligo",
    "Astro": "Astro", "Mic": "Microglia", "OPC": "OPC",
}
ALL_LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]
LABEL_DISPLAY = {
    "all_cells": "All cells", "Exc": "Excitatory", "Inh": "Inhibitory",
    "Oligo": "Oligodendrocyte", "Astro": "Astrocyte",
    "Mic": "Microglia", "OPC": "OPC",
}
CT_COLORS = {
    "all_cells": "#444444", "Exc": "#E64B35", "Inh": "#4DBBD5",
    "Oligo": "#00A087", "Astro": "#3C5488", "Mic": "#F39B7F", "OPC": "#8491B4",
}
GROUP_ORDER = ["High × F", "High × M", "Not High × F", "Not High × M"]
GROUP_PALETTE = {
    "High × F":     "#c0392b",
    "High × M":     "#2980b9",
    "Not High × F": "#e07b7b",
    "Not High × M": "#7baed4",
}


def zscore_per_donor(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def peak_contribution(Xte_raw, ridge_coef, peak_idx):
    """Compute z-scored contribution for a single peak without materialising all P contributions."""
    X = Xte_raw.astype(np.float64)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    sd[sd == 0] = 1.0
    z_p = (X[:, peak_idx] - mu) / sd          # (n,)
    return z_p * ridge_coef[peak_idx]


def main():
    # ── metadata ─────────────────────────────────────────────────────────────
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

    ov            = np.load(OVERLAP_NPZ, allow_pickle=True)
    valid_sea_idx = ov["valid_sea_idx"]

    print("Loading feature names ...")
    with open(PCT_CACHE, "rb") as f:
        cache = pickle.load(f)
    feat_names = cache["feat_names"]
    feat_genes = cache["feat_genes"]
    feat_index = {name: i for i, name in enumerate(feat_names)}

    print("Loading Ridge coefficient cache ...")
    with open(COEF_CACHE, "rb") as f:
        coef_cache = pickle.load(f)

    print("Loading SEA-AD pseudobulks ...")
    with open(SEA_AC_PKL, "rb") as f:
        sea_ac = pickle.load(f)
    sea_donors   = np.array(sea_ac["donors"])
    dlpfc_mask   = np.array([d in dlpfc43 for d in sea_donors])
    dlpfc_donors = sea_donors[dlpfc_mask]
    sea_ac_raw   = sea_ac["peak_sum"][dlpfc_mask][:, valid_sea_idx]
    del sea_ac; gc.collect()

    dlpfc_adnc = np.array([adnc_series.get(d) for d in dlpfc_donors])
    dlpfc_sex  = np.array([sex_series.get(d)  for d in dlpfc_donors])

    with open(SEA_CT_PKL, "rb") as f:
        sea_ct = pickle.load(f)

    # ── identify top peak per label ───────────────────────────────────────────
    print("Selecting top IUT+posslope peak per cell type ...")
    selections = {}   # label -> {peak, gene, pi, pval_iut, diff_HM, diff_NHF, diff_NHM, min_gap}

    for label in ALL_LABELS:
        df = pd.read_csv(RES / f"lm_contrib_peak_rankings_{label}.csv")

        # load donor vectors for this label
        if label == "all_cells":
            Xte_raw = sea_ac_raw
        else:
            Xte_raw = sea_ct[CT_KEY[label]][dlpfc_mask]

        is_high   = (dlpfc_adnc == "High")
        is_female = (dlpfc_sex  == "Female")
        donors    = dlpfc_donors.copy()

        nz      = (Xte_raw > 0).any(axis=1)
        Xte_sub = Xte_raw[nz]
        is_high_s   = is_high[nz]
        is_female_s = is_female[nz]

        ridge_coef = coef_cache[label].astype(np.float64)

        # load true ages
        res_path = RES / f"xds_pfc357_peaks_{label}_all.csv"
        df_age   = pd.read_csv(res_path).set_index("donor_id")["true_age"]
        true_age = np.array([df_age.get(d, np.nan) for d in donors[nz]])

        # compute HF slope for all candidate peaks
        candidates = df[df["min_gap"] > 0].sort_values("pval_iut")
        hf_valid   = is_high_s & is_female_s & ~np.isnan(true_age)
        age_hf     = true_age[hf_valid].astype(np.float64)
        age_hf_c   = age_hf - age_hf.mean()
        denom      = np.dot(age_hf_c, age_hf_c)

        top_row = None
        for _, row in candidates.iterrows():
            pi = feat_index.get(row["peak"])
            if pi is None:
                continue
            c_hf = peak_contribution(Xte_sub, ridge_coef, pi)[hf_valid]
            slope_hf = np.dot(age_hf_c, c_hf) / denom if denom > 0 else 0.0
            if slope_hf > 0:
                top_row = row
                break

        if top_row is None:
            print(f"  [{label}] WARNING: no positive-slope peak found, using rank-1 by pval_iut")
            top_row = candidates.iloc[0]

        pi   = feat_index[top_row["peak"]]
        gene = feat_genes[pi] if feat_genes[pi] != "?" else ""

        # precompute contribution for this peak
        contrib_peak = peak_contribution(Xte_sub, ridge_coef, pi)

        selections[label] = dict(
            peak=top_row["peak"], gene=gene, pi=pi,
            pval_iut=top_row["pval_iut"],
            diff_HM=top_row["diff_HM"],
            diff_NHF=top_row["diff_NHF"],
            diff_NHM=top_row["diff_NHM"],
            min_gap=top_row["min_gap"],
            contrib=contrib_peak,
            true_age=true_age,
            is_high=is_high_s,
            is_female=is_female_s,
        )
        print(f"  {label:12s}  top peak: {top_row['peak']}  gene={gene or '(intergenic)'}  "
              f"pval_iut={top_row['pval_iut']:.4f}")
        log_test(
            "seaad_alltissues_investigate", "dlpfc_top_iut_posslope_scatter_combined",
            analysis="IUT (intersection-union test) for peak ADNC-group ordering, top posslope peak",
            test="intersection_union_test", group_a=label, group_b=top_row["peak"],
            p_value=top_row["pval_iut"],
            notes=f"gene={gene}; diff_HM={top_row['diff_HM']}; min_gap={top_row['min_gap']}",
        )

    del sea_ct, sea_ac_raw; gc.collect()

    # ── figure ────────────────────────────────────────────────────────────────
    print("\nPlotting combined figure ...")
    n = len(ALL_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.6), squeeze=True)

    for ax, label in zip(axes, ALL_LABELS):
        sel      = selections[label]
        vals     = sel["contrib"]
        true_age = sel["true_age"]
        is_high  = sel["is_high"]
        is_female = sel["is_female"]

        masks = {
            "High × F":     is_high  &  is_female,
            "High × M":     is_high  & ~is_female,
            "Not High × F": ~is_high &  is_female,
            "Not High × M": ~is_high & ~is_female,
        }
        valid_age = ~np.isnan(true_age)
        is_hf     = is_high & is_female

        # scatter + trendlines
        for grp in GROUP_ORDER:
            m = masks[grp] & valid_age
            if m.sum() == 0:
                continue
            color  = GROUP_PALETTE[grp]
            age_g  = true_age[m]
            val_g  = vals[m]
            ax.scatter(age_g, val_g, color=color, s=28, alpha=0.85,
                       zorder=4, edgecolors="white", linewidths=0.3,
                       label=grp)
            if m.sum() >= 2:
                sl, ic, *_ = scipy_stats.linregress(age_g, val_g)
                x0, x1 = age_g.min(), age_g.max()
                ax.plot([x0, x1], [sl * x0 + ic, sl * x1 + ic],
                        color=color, lw=1.8, zorder=3)

        # pooled-others reference line
        m_other = ~is_hf & valid_age
        if m_other.sum() >= 2:
            age_oth = true_age[m_other]
            val_oth = vals[m_other]
            sl, ic, *_ = scipy_stats.linregress(age_oth, val_oth)
            x0_o = true_age[valid_age].min()
            x1_o = true_age[valid_age].max()
            ax.plot([x0_o, x1_o], [sl * x0_o + ic, sl * x1_o + ic],
                    color="grey", lw=1.2, ls="--", zorder=2,
                    label="Others (pooled)")

        # y-axis from data bounds
        scat_vals = np.concatenate([vals[masks[g] & valid_age]
                                    for g in GROUP_ORDER
                                    if (masks[g] & valid_age).sum() > 0])
        margin = max((scat_vals.max() - scat_vals.min()) * 0.10, 1e-6)
        ax.set_ylim(scat_vals.min() - margin, scat_vals.max() + margin)

        # annotation above axes
        piut  = sel["pval_iut"]
        dHM   = sel["diff_HM"];  dNHF = sel["diff_NHF"];  dNHM = sel["diff_NHM"]
        sig   = ("***" if piut < 0.001 else "**" if piut < 0.01
                 else "*" if piut < 0.05 else "ns")
        ann   = (f"IUT p={piut:.4f} {sig}\n"
                 f"HF−HM={dHM:+.4f}  HF−NHF={dNHF:+.4f}  HF−NHM={dNHM:+.4f}")
        ax.text(0.5, 1.02, ann,
                ha="center", va="bottom", fontsize=5.5,
                transform=ax.transAxes, clip_on=False,
                color="#c0392b" if piut < 0.05 else "#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, lw=0.4))

        # axis labels and title
        peak_short = sel["peak"].replace("chr", "")
        gene_str   = sel["gene"] or "(intergenic)"
        ct_color   = CT_COLORS[label]
        ax.set_title(f"{LABEL_DISPLAY[label]}\n{gene_str}  ({peak_short})",
                     fontsize=7.5, fontweight="bold", color=ct_color, pad=2)
        ax.set_xlabel("Age (years)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Clock contribution  (coeff × z-access)", fontsize=8)

    # shared legend on last panel
    handles, labels_leg = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels_leg, fontsize=5.5, loc="upper left",
                    framealpha=0.85, handlelength=1.2)

    fig.suptitle(
        "Top IUT peak (HF > all groups, positive HF slope)  per cell type\n"
        "Age vs. clock contribution  ·  trendlines per group  ·  dashed = pooled others",
        fontsize=9, y=1.12,
    )
    fig.tight_layout()
    outpath = FIG / "dlpfc_top_iut_posslope_scatter_combined.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {outpath.name}")


if __name__ == "__main__":
    main()
