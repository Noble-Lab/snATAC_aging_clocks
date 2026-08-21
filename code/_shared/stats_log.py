"""
Shared collector for statistical-test results and clock-coefficient tables.

Every project script that runs a stat test (Spearman/Pearson/Mann-Whitney/Cohen's d/
enrichment p-value/etc.) or fits a linear/tree clock should call `log_test(...)` or
`log_coef(...)` below. Each call appends one row to a per-project CSV under
outputs/_raw/<project>_statistical_tests.csv / _clock_coefficients.csv in the repo
root; code/_shared/merge_outputs.py (run once at the end) splits every project's
raw CSVs into one file per (project, analysis) under outputs/statistical_tests/
and one file per (project, clock_name) under outputs/clock_coefficients/.
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR   = REPO_ROOT / "outputs" / "_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_TEST_COLS = ["project", "figure", "analysis", "test", "group_a", "group_b",
              "n_a", "n_b", "statistic", "effect_size", "p_value", "q_value", "notes"]
_COEF_COLS = ["project", "clock_name", "modality", "cell_type", "feature",
              "coefficient", "importance", "rank", "model_type"]


def _append(path, cols, row):
    row = {k: row.get(k, "") for k in cols}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def log_test(project, figure, analysis, test, **kw):
    """Log one statistical test result. kw may include group_a, group_b, n_a, n_b,
    statistic, effect_size, p_value, q_value, notes."""
    row = dict(project=project, figure=figure, analysis=analysis, test=test, **kw)
    _append(RAW_DIR / f"{project}_statistical_tests.csv", _TEST_COLS, row)


def log_coef(project, clock_name, feature, coefficient, **kw):
    """Log one clock coefficient / feature-importance row. kw may include modality,
    cell_type, importance, rank, model_type."""
    row = dict(project=project, clock_name=clock_name, feature=feature,
               coefficient=coefficient, **kw)
    _append(RAW_DIR / f"{project}_clock_coefficients.csv", _COEF_COLS, row)
