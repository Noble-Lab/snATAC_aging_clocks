"""
Run AME (scoring=max, method=spearman) for the top-100 SHAP-ranked peaks per
cell type, against the JASPAR 2026 and HOCOMOCOv14 motif databases.

Requires FASTA files already written by meme_chip_sweep.py
(results/meme_chip_runs/fastas/fg_{label}_top100.fa).

Writes results/ame_max_scoring_runs/spearman_{db}_{label}_top100/ame.tsv,
one per (motif database, cell type). ame_max_scoring_plot.py parses these
TSVs, applies BH correction, and produces the figure.
"""

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FA_DIR  = RESULTS / "meme_chip_runs" / "fastas"
OUT_DIR = RESULTS / "ame_max_scoring_runs"
AME_BIN = "~/miniconda3/envs/reprod/bin/ame"

MOTIF_DBS = {
    "jaspar2026":  ROOT / "motif_databases" / "JASPAR2026_CORE_vertebrates_non-redundant.meme",
    "hocomocov14": ROOT / "motif_databases" / "HOCOMOCOv14_CORE_HUMAN.meme",
}
ALL_LABELS = ["All_cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]
TOP_N      = 100
N_PARALLEL = 6

OUT_DIR.mkdir(parents=True, exist_ok=True)

for dbname, dbpath in MOTIF_DBS.items():
    assert dbpath.exists(), f"Missing DB: {dbpath}"
for label in ALL_LABELS:
    fa = FA_DIR / f"fg_{label}_top{TOP_N}.fa"
    assert fa.exists(), f"Missing FASTA: {fa}"


def run_ame_job(args):
    dbname, label = args
    fg_fa   = FA_DIR / f"fg_{label}_top{TOP_N}.fa"
    out_dir = OUT_DIR / f"spearman_{dbname}_{label}_top{TOP_N}"
    tsv     = out_dir / "ame.tsv"
    if tsv.exists() and tsv.stat().st_size > 0:
        return dbname, label, "cached"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        AME_BIN, "--text",
        "--method",  "spearman",
        "--scoring", "max",
        str(fg_fa),
        str(MOTIF_DBS[dbname]),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.stdout.strip():
            tsv.write_text(r.stdout)
            return dbname, label, "ok"
        return dbname, label, f"empty\n{r.stderr[-300:]}"
    except subprocess.TimeoutExpired:
        return dbname, label, "timeout"
    except Exception as e:
        return dbname, label, f"error:{e}"


def main():
    jobs = [(dbname, label) for dbname in MOTIF_DBS for label in ALL_LABELS]
    print(f"Running {len(jobs)} AME jobs ({N_PARALLEL} parallel)…", flush=True)
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as pool:
        futs = {pool.submit(run_ame_job, j): j for j in jobs}
        for fut in as_completed(futs):
            dbname, label, status = fut.result()
            tag = "WARN" if status not in ("cached", "ok") else "    "
            print(f"  {tag}[{dbname}] {label}: {status}", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
