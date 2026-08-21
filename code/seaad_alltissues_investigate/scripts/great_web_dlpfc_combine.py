"""Combine the per-(label, tag) raw MSigDB-Pathway tables fetched from the
REAL GREAT web server (great.stanford.edu, v3.0.0/hg19) into pooled,
BH-corrected REACTOME-only CSVs. Mirrors
~/pfc_internal_valid/scripts/great_web_combine.py.

Loads: results/great_web_dlpfc_raw/{label}__{tag}.csv
Saves: results/dlpfc_pathway_enrichment_GREATweb_Reactome_{tag}.csv
"""
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW_DIR = RESULTS / "great_web_dlpfc_raw"

ALL_LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]


def discover_tags() -> list[str]:
    """{label}__{tag}.csv -> unique sorted tags, auto-discovered from
    whatever raw files great_web_dlpfc_sweep.R has produced so far."""
    tags = set()
    for f in RAW_DIR.glob("*__*.csv"):
        label, tag = f.stem.split("__", 1)
        if label in ALL_LABELS:
            tags.add(tag)
    return sorted(tags)


def main():
    tags = discover_tags()
    print(f"Discovered {len(tags)} tags: {tags}")
    for tag in tags:
        rows = []
        for label in ALL_LABELS:
            f = RAW_DIR / f"{label}__{tag}.csv"
            if not f.exists():
                print(f"  missing: {f.name}")
                continue
            df = pd.read_csv(f)
            df["label"] = label
            rows.append(df)

        if not rows:
            print(f"{tag}: no raw files found, skipping")
            continue

        df_all = pd.concat(rows, ignore_index=True)
        df_reactome = df_all[df_all["ID"].str.startswith("REACTOME_")].copy()
        if df_reactome.empty:
            print(f"{tag}: 0 REACTOME_* terms, skipping")
            continue

        # BH pooled across all 7 labels x REACTOME_* terms only (recomputed
        # here, since Reactome's multiple-testing burden differs from the
        # full MSigDB Pathway collection GREAT tested against).
        _, q_binom, _, _ = multipletests(df_reactome["Binom_Raw_PValue"].astype(float), method="fdr_bh")
        _, q_hyper, _, _ = multipletests(df_reactome["Hyper_Raw_PValue"].astype(float), method="fdr_bh")
        df_reactome["q_binom"] = q_binom
        df_reactome["q_hypergeom"] = q_hyper

        out = RESULTS / f"dlpfc_pathway_enrichment_GREATweb_Reactome_{tag}.csv"
        df_reactome.to_csv(out, index=False)
        n_sig = (df_reactome["q_binom"] < 0.05).sum()
        print(f"{tag}: saved {out.name}  ({len(df_reactome):,} tests, {n_sig} q_binom<0.05)")

    print("\nDone.")


if __name__ == "__main__":
    main()
