"""
Plot AME motif enrichment for the top-100 SHAP-ranked peaks per cell type,
against the JASPAR 2026 and HOCOMOCOv14 motif databases.

For each motif, "enrichment" is the Spearman correlation (from AME's
scoring=max Spearman test) between the motif's max score in each peak and
the peak's SHAP rank, tested per (cell type, motif database) and BH-corrected
globally across all 7 cell types.

Reads AME TSV output from ame_max_scoring_sweep.py (results/ame_max_scoring_runs/).
Saves figures/ame_max_scoring/ame_max_spearman_combined_top100.pdf.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

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
import matplotlib.patches
from statsmodels.stats.multitest import multipletests

mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "ame_max_scoring_runs"
FIG_DIR = ROOT / "figures" / "ame_max_scoring"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MOTIF_DBS  = ["jaspar2026", "hocomocov14"]
ALL_LABELS = ["All_cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
TOP_N      = 100
MAX_PER_CT = 5

CT_COLORS = {
    "All_cells":  "#000000", "Excitatory": "#E87722",
    "Inhibitory": "#2CA02C", "Oligo":      "#1F77B4",
    "Astro":      "#D62728", "Microglia":  "#9467BD",
    "OPC":        "#8C564B",
}
CT_ORDER = ["All_cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


def normalize_gene(gene):
    gene = re.sub(r'^ZN(\d)', r'ZNF\1', gene)
    gene = re.sub(r'^PO(\d)', r'POU\1', gene)
    return gene


def gene_from_motif_id(mid):
    if ".H14CORE" in mid:
        return mid.split(".H14CORE")[0]
    parts = re.split(r"[_\.\|]", str(mid))
    g = parts[0]
    if len(g) < 3 and len(parts) > 1:
        g = "_".join(parts[:2])
    return g[:18]


def parse_ame_tsv(tsv_path, dbname, label):
    """Parse one AME spearman-scoring TSV into rows of {gene, raw_pval, spearman_r}."""
    rows = []
    if not tsv_path.exists():
        return rows
    header = None
    with open(tsv_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            r = dict(zip(header, cols))
            try:
                mid      = r.get("motif_ID", "")
                alt_id   = r.get("motif_alt_ID", "").strip()
                raw_gene = alt_id if alt_id else gene_from_motif_id(mid)
                gene     = normalize_gene(raw_gene)
                raw_p    = float(r.get("p-value", 1.0))
                try:
                    spearman_r = float(r.get("Spearmans_CC", ""))
                except (ValueError, TypeError):
                    spearman_r = float("nan")
                rows.append({
                    "db": dbname, "condition": label, "motif_id": mid,
                    "gene": gene, "raw_pval": raw_p, "spearman_r": spearman_r,
                })
            except Exception:
                pass
    return rows


def make_panel(ax, df, title):
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#aaa")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    sig = df[df["global_q"] < 0.05].copy()
    n_sig_total = len(sig)

    parts = []
    for ct in CT_ORDER:
        sub = (sig[sig["condition"] == ct]
               .sort_values("global_q")
               .drop_duplicates("gene", keep="first")
               .head(MAX_PER_CT))
        parts.append(sub)
    sig = pd.concat(parts, ignore_index=True)

    if sig.empty:
        ax.text(0.5, 0.5, "no sig. motifs\n(q ≥ 0.05)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#aaa")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    sig = sig.iloc[::-1].reset_index(drop=True)
    colors = [CT_COLORS.get(c, "#888") for c in sig["condition"]]

    ax.barh(range(len(sig)), sig["neglog_q"], color=colors, alpha=0.85,
            height=0.65, edgecolor="none")
    max_x = sig["neglog_q"].max() or 1.0

    for i, (_, row) in enumerate(sig.iterrows()):
        r = row.get("spearman_r", float("nan"))
        if not np.isnan(r):
            ax.text(0.15, i, f"r={r:.2f}", va="center", ha="left",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["gene"], fontsize=9)
    ax.axvline(-np.log10(0.05), color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(0, max_x * 1.3)
    ax.set_xlabel("−log₁₀(global q)", fontsize=9)
    ax.set_title(f"{title}\n(≤{MAX_PER_CT}/CT shown of {n_sig_total} sig.)",
                 fontsize=10, fontweight="bold")


def main():
    print("Parsing TSVs…", flush=True)
    raw_by_db = defaultdict(list)
    for dbname in MOTIF_DBS:
        for label in ALL_LABELS:
            tsv = OUT_DIR / f"spearman_{dbname}_{label}_top{TOP_N}" / "ame.tsv"
            raw_by_db[dbname].extend(parse_ame_tsv(tsv, dbname, label))

    print("Applying global BH…", flush=True)
    corrected = {}
    for dbname, rows in sorted(raw_by_db.items()):
        if not rows:
            continue
        df    = pd.DataFrame(rows)
        pvals = df["raw_pval"].clip(lower=1e-300).values
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        df["global_q"] = qvals
        df["neglog_q"] = -np.log10(np.clip(qvals, 1e-300, 1))
        corrected[dbname] = df
        n_sig = (df["global_q"] < 0.05).sum()
        print(f"  {dbname} top{TOP_N}: {n_sig} sig", flush=True)

        for _, row in df.iterrows():
            log_test(
                "pfc_internal_valid", "ame_max_spearman_combined_top100",
                analysis=f"AME max-scoring motif enrichment ({dbname})",
                test="spearman_correlation_vs_shap_rank",
                group_a=row["condition"], group_b=row["motif_id"],
                statistic=row["spearman_r"], p_value=row["raw_pval"], q_value=row["global_q"],
                notes=f"gene={row['gene']}; db={dbname}; top_n={TOP_N}",
            )

    print("\nGenerating figure…", flush=True)
    df_j = corrected.get("jaspar2026",  pd.DataFrame())
    df_h = corrected.get("hocomocov14", pd.DataFrame())

    n_j = int((df_j["global_q"] < 0.05).sum()) if not df_j.empty else 0
    n_h = int((df_h["global_q"] < 0.05).sum()) if not df_h.empty else 0

    shown_j  = min(n_j, len(CT_ORDER) * MAX_PER_CT)
    shown_h  = min(n_h, len(CT_ORDER) * MAX_PER_CT)
    max_rows = max(shown_j, shown_h, 4)
    fig_h    = max(max_rows * 0.38 + 2.8, 5)

    fig, (ax_j, ax_h) = plt.subplots(1, 2, figsize=(14, fig_h),
                                      constrained_layout=True)
    make_panel(ax_j, df_j, f"JASPAR 2026  (n={n_j})")
    make_panel(ax_h, df_h, f"HOCOMOCOv14  (n={n_h})")

    handles = [
        mpl.patches.Patch(color=CT_COLORS[c], label=c.replace("_", " "))
        for c in CT_ORDER
    ]
    fig.legend(handles=handles, title="Cell type", fontsize=9, title_fontsize=9.5,
               loc="lower center", ncol=len(CT_ORDER),
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        f"AME motif enrichment — spearman (scoring=max) — top-{TOP_N} SHAP peaks\n"
        f"Global BH FDR across all 7 cell types  (q < 0.05)  |  bar labels: Spearman r",
        fontsize=10,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"ame_max_spearman_combined_top{TOP_N}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  top{TOP_N}: JASPAR n={n_j}, HOCOMOCO n={n_h}", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
