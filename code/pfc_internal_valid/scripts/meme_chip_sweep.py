"""
Write FASTA files of the top-100 SHAP-ranked peak sequences per cell type, for
ame_max_scoring_sweep.py's AME motif-enrichment run.

Peak sequences are pulled from the hg38 2bit genome and cached
(results/motif_sweep_cache/seq_cache.pkl) the first time this runs.

Saves: results/meme_chip_runs/fastas/fg_{condition}_top100.fa
"""

import pickle, random
from pathlib import Path

import numpy as np
import pandas as pd
import twobitreader

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RESULTS   = ROOT / "results"
MEME_DIR  = RESULTS / "meme_chip_runs"
FASTA_DIR = MEME_DIR / "fastas"
FASTA_DIR.mkdir(parents=True, exist_ok=True)

GENOME_2BIT = RESULTS / "hg38.2bit"
SEQ_CACHE   = RESULTS / "motif_sweep_cache" / "seq_cache.pkl"

TOP_N = 100

ALL_LABELS = ["All cells", "Excitatory", "Inhibitory", "Oligo", "Astro", "Microglia", "OPC"]


def write_fasta(peak_list, path):
    with open(path, "w") as f:
        for i, p in enumerate(peak_list):
            seq = seq_cache.get(p)
            if seq and len(seq) >= 50 and seq.count("N") < len(seq) * 0.5:
                f.write(f">seq{i}  {p}\n{seq}\n")


print("Loading SHAP values…", flush=True)
with open(RESULTS / "shap_mean_abs.pkl", "rb") as f:
    shap = pickle.load(f)
peak_map = pd.read_csv(RESULTS / "peak_gene_map.csv")
idx2peak = dict(zip(peak_map["peak_idx"], peak_map["peak_name"]))

print("Loading sequence cache…", flush=True)
if SEQ_CACHE.exists():
    with open(SEQ_CACHE, "rb") as f:
        seq_cache = pickle.load(f)
else:
    genome = twobitreader.TwoBitFile(str(GENOME_2BIT))
    all_fg = set()
    for label in ALL_LABELS:
        s = np.array(shap[label])
        for i in np.argsort(s)[::-1][:2000]:
            if i in idx2peak:
                all_fg.add(i)
    bg_pool = [i for i in peak_map["peak_idx"] if i not in all_fg and i in idx2peak]
    random.seed(42)
    bg_sample = {idx2peak[i] for i in random.sample(bg_pool, 5000)}
    all_peaks = [idx2peak[i] for i in all_fg] + list(bg_sample)
    seq_cache = {}
    for pk in all_peaks:
        try:
            ch, co = pk.split(":"); st, en = co.split("-")
            seq_cache[pk] = genome[ch][int(st):int(en)].upper()
        except Exception:
            seq_cache[pk] = None
    SEQ_CACHE.parent.mkdir(exist_ok=True)
    with open(SEQ_CACHE, "wb") as f:
        pickle.dump(seq_cache, f)

fg_peaks = {}
for label in ALL_LABELS:
    s   = np.array(shap[label])
    top = np.argsort(s)[::-1][:2000]
    fg_peaks[label] = [idx2peak[i] for i in top if i in idx2peak]

print("Writing FASTA files…", flush=True)
n_written = 0
for label in ALL_LABELS:
    safe = label.replace(" ", "_")
    fa_path = FASTA_DIR / f"fg_{safe}_top{TOP_N}.fa"
    write_fasta(fg_peaks[label][:TOP_N], fa_path)
    n_written += 1

print(f"  {n_written} FASTA files ready", flush=True)
print("Done.")
