"""Plot residual vs ADNC for the peaks/all feature regime (all-cells + per-CT).

For each combo, produces:
  dlpfc_xds_pfc357_peaks_{label}_all_adnc_violin.pdf
      violin of OLS-corrected residual by ADNC category
  dlpfc_xds_pfc357_peaks_{label}_all_adnc_scatter.pdf
      scatter of true vs predicted age coloured by ADNC

Also prints Spearman rho (residual vs ADNC) for each combo.
"""
from __future__ import annotations

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RES  = ROOT / "results"
FIG  = ROOT / "figures"

SEAAD_META = Path("~/age_accel_per_cell_type/cache/seaad_pseudobulks_by_ct.pkl")

ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
ADNC_NUM   = {v: i for i, v in enumerate(ADNC_ORDER)}
ADNC_PALETTE = {
    "Not AD":       "#4393c3",
    "Low":          "#92c5de",
    "Intermediate": "#f4a582",
    "High":         "#b2182b",
}

LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]


def load_with_adnc(label: str, braak_series: pd.Series,
                   adnc_series: pd.Series) -> pd.DataFrame | None:
    path = RES / f"xds_pfc357_peaks_{label}_all.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["adnc"]     = df["donor_id"].map(adnc_series)
    df["adnc_num"] = df["adnc"].map(ADNC_NUM)
    return df


def plot_adnc_violin(df: pd.DataFrame, title: str, outpath: Path) -> None:
    stages = [s for s in ADNC_ORDER if s in df["adnc"].values]
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(stages)), 4.5))

    data_by_stage = [df.loc[df["adnc"] == st, "residual"].dropna().values
                     for st in stages]
    parts = ax.violinplot(data_by_stage, positions=range(len(stages)),
                          showmedians=True, showextrema=True)
    for pc, st in zip(parts["bodies"], stages):
        pc.set_facecolor(ADNC_PALETTE.get(st, "#adb5bd"))
        pc.set_alpha(0.65)

    for i, (stage, grp) in enumerate(zip(stages, data_by_stage)):
        jitter = np.random.default_rng(i).uniform(-0.07, 0.07, len(grp))
        ax.scatter(i + jitter, grp, color=ADNC_PALETTE.get(stage, "#adb5bd"),
                   s=20, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.3)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("OLS-corrected brain-age residual (years)", fontsize=9)

    nums  = df["adnc_num"].values.astype(float)
    valid = ~np.isnan(nums)
    if valid.sum() >= 3:
        rho, pval = stats.spearmanr(nums[valid], df["residual"].values[valid])
        ax.set_title(f"{title}\nSpearman ρ={rho:+.3f}  p={pval:.3g}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    for i, (st, grp) in enumerate(zip(stages, data_by_stage)):
        ax.text(i, ax.get_ylim()[0] + 0.5, f"n={len(grp)}",
                ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def main() -> None:
    # The donor-metadata cache was pickled under numpy>=2.0; _seaad_meta_compat.py loads
    # a numpy-version-agnostic copy of it instead of unpickling it directly.
    from _seaad_meta_compat import load_seaad_meta
    meta = load_seaad_meta()
    braak_series = meta["Braak"]
    adnc_series  = meta["Overall AD neuropathological Change"]

    print("ADNC distribution (all SEA-AD):")
    print(adnc_series.value_counts())

    result_rows = []
    for label in LABELS:
        df = load_with_adnc(label, braak_series, adnc_series)
        if df is None or df.empty:
            print(f"  [{label}] no CSV found, skipping")
            continue

        print(f"\n[{label}]  n={len(df)}")
        print("  ADNC dist:", df["adnc"].value_counts().to_dict())

        # violin
        plot_adnc_violin(
            df,
            title=f"PFC-357 peaks → DLPFC  [{label}/all]\nresidual vs ADNC",
            outpath=FIG / f"dlpfc_xds_pfc357_peaks_{label}_all_adnc_violin.pdf",
        )
        # Spearman rho
        nums  = df["adnc_num"].values.astype(float)
        valid = ~np.isnan(nums)
        rho, pval = stats.spearmanr(nums[valid], df["residual"].values[valid])
        pred_r = float(stats.pearsonr(df["true_age"], df["pred_age"])[0])
        result_rows.append({
            "label": label, "n": int(valid.sum()),
            "pred_r": pred_r, "adnc_rho": rho, "adnc_p": pval,
        })
        log_test(
            "seaad_alltissues_investigate", f"dlpfc_xds_pfc357_peaks_{label}_all_adnc_violin",
            analysis="PFC-357-peak DLPFC age-clock residual vs ADNC severity",
            test="spearman_correlation", group_a=label, n_a=int(valid.sum()),
            statistic=rho, p_value=pval, notes=f"pred_r={pred_r:.4f}",
        )

    print("\n\n══ Residual vs ADNC (peaks/all) ══")
    tbl = pd.DataFrame(result_rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(RES / "xds_pfc357_peaks_adnc_correlation.csv", index=False)
    print("\nwrote xds_pfc357_peaks_adnc_correlation.csv")


if __name__ == "__main__":
    main()
