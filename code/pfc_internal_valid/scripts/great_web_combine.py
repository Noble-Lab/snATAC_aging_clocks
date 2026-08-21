"""
Combine the per-(label, top_n) raw enrichment tables fetched from the REAL
GREAT web server (great.stanford.edu, v3.0.0/hg19) into pooled, BH-corrected
CSVs (BH correction applied per top_n, jointly across all cell types).

Two sources:
  - results/great_web_raw/{label}_top{N}.csv           "MSigDB Pathway" table
    (great_web_sweep.R) -- a mixed collection of REACTOME_*, KEGG_*,
    BIOCARTA_*, PID_* (and a few smaller ST_/SA_/SIG_ sets) gene sets, split
    out here into per-source-database CSVs.
  - results/great_web_raw_<slug>/{label}_top{N}.csv     other ontologies
    (great_web_sweep_multiontology.R): GO Biological Process/Molecular
    Function/Cellular Component, PANTHER Pathway, BioCyc Pathway.

Saves (per top_n): results/pathway_enrichment_GREATweb_<name>_top{N}.csv
  <name> in: MSigDBPathway, Reactome, KEGG, BioCarta, PID,
             GO_BP, GO_MF, GO_CC, PANTHER, BioCyc
"""
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ALL_LABELS = ["All cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
TOP_N_LIST = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [2000]

# (output_name, raw_dir_relative_to_RESULTS, optional ID-prefix filter)
SOURCES = [
    ("MSigDBPathway", "great_web_raw", None),
    ("Reactome",      "great_web_raw", "REACTOME_"),
    ("KEGG",          "great_web_raw", "KEGG_"),
    ("BioCarta",      "great_web_raw", "BIOCARTA_"),
    ("PID",           "great_web_raw", "PID_"),
    ("GO_BP",         "great_web_raw_go_bp",   None),
    ("GO_MF",         "great_web_raw_go_mf",   None),
    ("GO_CC",         "great_web_raw_go_cc",   None),
    ("PANTHER",       "great_web_raw_panther", None),
    ("BioCyc",        "great_web_raw_biocyc",  None),
]

def load_raw(raw_dir, top_n):
    rows = []
    for label in ALL_LABELS:
        f = RESULTS / raw_dir / f"{label.replace(' ', '_')}_top{top_n}.csv"
        if not f.exists():
            continue
        rows.append(pd.read_csv(f))
    return pd.concat(rows, ignore_index=True) if rows else None

for top_n in TOP_N_LIST:
    print(f"\n=== top{top_n} ===")
    combined_cache = {}
    for name, raw_dir, prefix in SOURCES:
        if raw_dir not in combined_cache:
            combined_cache[raw_dir] = load_raw(raw_dir, top_n)
        df_all = combined_cache[raw_dir]
        if df_all is None:
            print(f"  {name}: no raw files found in {raw_dir}, skipping")
            continue

        df = df_all[df_all["ID"].str.startswith(prefix)].copy() if prefix else df_all.copy()
        if df.empty:
            print(f"  {name}: 0 terms after filtering, skipping")
            continue

        # BH pooled across all 7 labels x this term set (recomputed per subset,
        # since e.g. KEGG's multiple-testing burden differs from all of MSigDB Pathway)
        _, q_binom, _, _ = multipletests(df["Binom_Raw_PValue"].astype(float), method="fdr_bh")
        _, q_hyper, _, _ = multipletests(df["Hyper_Raw_PValue"].astype(float), method="fdr_bh")
        df["q_binom"] = q_binom
        df["q_hypergeom"] = q_hyper

        out = RESULTS / f"pathway_enrichment_GREATweb_{name}_top{top_n}.csv"
        df.to_csv(out, index=False)
        n_sig = (df["q_binom"] < 0.05).sum()
        print(f"  {name}: saved {out.name}  ({len(df):,} tests, {n_sig} q_binom<0.05)")

print("\nDone.")
