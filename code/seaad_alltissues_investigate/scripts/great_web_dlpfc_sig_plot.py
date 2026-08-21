"""
Shared constants and pathway-name helpers for the DLPFC GREAT-web Reactome
enrichment figures, used by great_web_dlpfc_top5_plot.py.
"""
import html
import re
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG     = ROOT / "figures"

ALL_LABELS = ["all_cells", "Exc", "Inh", "Oligo", "Astro", "Mic", "OPC"]
LABEL_DISPLAY = {
    "all_cells": "All cells", "Exc": "Excitatory", "Inh": "Inhibitory",
    "Oligo": "Oligo", "Astro": "Astro", "Mic": "Microglia", "OPC": "OPC",
}
CT_COLORS = {
    "all_cells": "#444444", "Exc": "#E64B35", "Inh": "#4DBBD5",
    "Oligo": "#00A087", "Astro": "#3C5488", "Mic": "#F39B7F", "OPC": "#8491B4",
}

_PREFIX_RE   = re.compile(r"^Genes involved in\s+", re.IGNORECASE)
_SHAPTOP_RE  = re.compile(r"^shaptop(\d+)$")
_SHAPDIFF_RE = re.compile(r"^shapdiffpct_top(\d+)_top(\d+)$")


def clean_name(name):
    return html.unescape(_PREFIX_RE.sub("", str(name)))


def describe_tag(tag: str) -> str:
    m = _SHAPTOP_RE.match(tag)
    if m:
        n_peaks = m.group(1)
        return f"Top-{n_peaks} peaks by mean |SHAP| (all DLPFC donors)"
    m = _SHAPDIFF_RE.match(tag)
    if m:
        n_donors, n_peaks = m.groups()
        return (f"Top-{n_peaks} peaks by Δ percentile-rank mean |SHAP|\n"
                f"(top-{n_donors} vs bottom-{n_donors} OLS-residual donors)")
    return tag
