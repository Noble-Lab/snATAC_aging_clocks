"""Split every project's raw stats/coefficient CSVs (outputs/_raw/*) into the
per-analysis deliverables under outputs/statistical_tests/ and
outputs/clock_coefficients/ -- one CSV per distinct (project, analysis) or
(project, clock_name), rather than one big file with an analysis/clock_name
column mixing every experiment together. Run this last, after all 8 projects
have executed.
"""
import re
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR   = REPO_ROOT / "outputs" / "_raw"
OUT_DIR   = REPO_ROOT / "outputs"


def _slug(s: str) -> str:
    """Filesystem-safe slug: lowercase, non-alnum runs -> single underscore."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_").lower()
    return s[:80] if len(s) > 80 else s


def split(pattern, out_subdir, key_cols):
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=False)
    df = df.drop_duplicates()

    dest = OUT_DIR / out_subdir
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.glob("*.csv"):
        f.unlink()  # clear stale per-analysis files from a prior run's different split

    n_files = 0
    for keys, sub in df.groupby(key_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        name = "__".join(_slug(k) for k in keys) + ".csv"
        sub.to_csv(dest / name, index=False)
        n_files += 1
    print(f"Wrote {n_files} files to {dest}/  ({len(df):,} rows total from {len(files)} raw files)")


if __name__ == "__main__":
    split("*_statistical_tests.csv", "statistical_tests", ["project", "analysis"])
    split("*_clock_coefficients.csv", "clock_coefficients", ["project", "clock_name"])
