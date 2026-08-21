"""
Plot all REACTOME_* pathways significant (q_binom < 0.05) from the REAL
GREAT web server (great.stanford.edu, v3.0.0/hg19) enrichment run.

- Pathway names ("Genes involved in " stripped) + pathway gene count (K) on
  the LEFT as tick labels
- No gap between cell-type groups
- Bars colored by cell type with the same CT color scheme

Usage:
    python great_web_sig.py [top_n]   (default: 2000)

Saves: figures/pathway_enrichment_GREATweb_sig_top{N}.pdf/.png
"""
import re
import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent.parent.parent / "_shared",
              Path("~/reproducability_expts_minimal/code/_shared")):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from stats_log import log_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG     = ROOT / "figures"

ALL_LABELS = ["All cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]

CT_COLORS = {
    "All cells":  "black",
    "Excitatory": "tab:orange",
    "Inhibitory": "tab:green",
    "Oligo":      "tab:blue",
    "Astro":      "tab:red",
    "Microglia":  "tab:purple",
    "OPC":        "tab:brown",
}

_PREFIX_RE = re.compile(r"^Genes involved in\s+", re.IGNORECASE)

def clean_name(name):
    return _PREFIX_RE.sub("", str(name))


def make_figure(top_n):
    csv_path = RESULTS / f"pathway_enrichment_GREATweb_Reactome_top{top_n}.csv"
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for _, r in df.iterrows():
        log_test(
            "pfc_internal_valid", f"pathway_enrichment_GREATweb_sig_top{top_n}",
            analysis="GREAT web (great.stanford.edu v3.0.0/hg19) REACTOME enrichment",
            test="binomial_region_enrichment",
            group_a=r["label"], group_b=r["name"],
            n_a=r.get("n_regions"), statistic=r.get("Binom_Fold_Enrichment"),
            p_value=r.get("Binom_Raw_PValue"), q_value=r.get("q_binom"),
            notes=f"top_n={top_n}; Hyper_Total_Genes={r.get('Hyper_Total_Genes')}",
        )

    sig = df[df["q_binom"] < 0.05].copy()
    print(f"\ntop{top_n}: {len(sig)} significant REACTOME_* pathways (q_binom < 0.05, real GREAT web server)")
    if sig.empty:
        print("  Nothing to plot.")
        return
    print(sig.groupby("label").size().to_string())

    sig["name_clean"] = sig["name"].apply(clean_name)
    sig["label_text"] = sig["name_clean"] + " (" + sig["Hyper_Total_Genes"].astype(int).astype(str) + ")"
    sig["neglog_q"]   = -np.log10(sig["q_binom"].clip(lower=1e-300))

    rows = []
    for label in ALL_LABELS:
        sub = sig[sig["label"] == label].sort_values("neglog_q", ascending=False)
        for _, r in sub.iterrows():
            rows.append({
                "label":      label,
                "label_text": r["label_text"],
                "neglog_q":   r["neglog_q"],
                "color":      CT_COLORS[label],
            })
    if not rows:
        return
    plot_df = pd.DataFrame(rows).reset_index(drop=True)

    n_bars  = len(plot_df)
    bar_h   = 0.32
    fig_h   = max(3.0, n_bars * bar_h + 1.8)
    max_x   = plot_df["neglog_q"].max()

    fig, ax = plt.subplots(figsize=(9, fig_h))
    y_pos = np.arange(n_bars, dtype=float)

    ax.barh(y_pos, plot_df["neglog_q"].values,
            color=plot_df["color"].values,
            height=bar_h * 0.82, alpha=0.85, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label_text"].values, fontsize=7.5)
    for ticklbl, color in zip(ax.get_yticklabels(), plot_df["color"].values):
        ticklbl.set_color(color)

    ax.axvline(-np.log10(0.05), color="grey", lw=0.9, ls="--", alpha=0.55)

    ax.set_xlim(0, max_x * 1.12)
    ax.set_ylim(-0.5, n_bars - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(q, cross-CT pooled BH, GREAT web server binomial test)", fontsize=9)
    ax.set_title(
        f"Real GREAT (great.stanford.edu, v3.0.0/hg19) — MSigDB Pathway: REACTOME — q < 0.05\n"
        f"Top {top_n} SHAP-ranked peaks per cell type",
        fontsize=10,
    )

    active_labels = plot_df["label"].unique()
    legend_handles = [
        mpatches.Patch(color=CT_COLORS[lbl], label=lbl)
        for lbl in ALL_LABELS if lbl in active_labels
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8,
              loc="lower right", title="Cell type", title_fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG / f"pathway_enrichment_GREATweb_sig_top{top_n}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    top_ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [2000]
    for n in top_ns:
        make_figure(n)
    print("Done.")
