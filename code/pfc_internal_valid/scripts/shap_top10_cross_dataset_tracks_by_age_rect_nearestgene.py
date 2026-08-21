"""
For each of the 7 SHAP top-10 peaks that agree in sign across PFC,
hippocampus, and mouse hypothalamus (from shap_enrichment_crossdataset.py),
plot a +/-150kb genome track: the peak's accessibility per age bucket
(z-scored against its own cross-donor mean/SD) as bar height, alongside
nearby genes and their distance to the peak.

Saves: figures/shap_top10_cross_dataset_tracks_by_age_rect_150kb_nearestgene.pdf/.png

External large-file dependencies (kept at fixed absolute paths, not inside
this repo):
  ~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl (238MB)
  ~/pfc_to_hippocampus/cache/hip_peak_pseudobulks_in_pfc_space.pkl (99MB)
The mouse hypothalamus pseudobulk/liftover caches are read from pfc_to_hypo's
own cache/ (code/pfc_to_hypo/cache/), which is in-repo.
"""
import pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from scipy.stats import pearsonr
import pyensembl

warnings.filterwarnings("ignore")
mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG     = ROOT / "figures"
FIG.mkdir(exist_ok=True)

AGE_COLORS = {"Y": "#4575b4", "M": "#fdae61", "A": "#d73027"}
AGE_ROW_LABELS = {"Y": "Young", "M": "Middle", "A": "Old"}
UP_C, DOWN_C = "#d73027", "#4575b4"     # target-peak highlight by sign of r
GENE_C       = "#2b2b2b"

HALF_WIDTH, WIDTH_LABEL = 150_000, "150kb"

# ── 1. filter SHAP top-10 peaks to "all 3 cohorts, same sign" ─────────────────
print("Loading SHAP top-10 peaks + cross-dataset correlations…")
shap_df = pd.read_csv(RESULTS / "shap_top10_peaks.csv")
cdc     = pd.read_csv(RESULTS / "cross_dataset_corr.csv")

merged = shap_df.merge(cdc[["peak_name", "r_pfc", "r_hip", "r_hypo", "n_datasets"]],
                        on="peak_name", how="left")
qual = merged[merged["n_datasets"] == 3].copy()
same_sign = (np.sign(qual["r_pfc"]) == np.sign(qual["r_hip"])) & \
            (np.sign(qual["r_hip"]) == np.sign(qual["r_hypo"]))
qual = qual[same_sign]

regions = (qual.groupby(["peak_name", "gene", "r_pfc", "r_hip", "r_hypo"], as_index=False)
               .agg(labels=("label", lambda s: ", ".join(sorted(set(s))))))
regions = regions.sort_values("r_pfc", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
print(f"{len(regions)} peaks pass the all-3-cohort / same-sign filter:")
for _, r in regions.iterrows():
    print(f"  {r['peak_name']:26s} {r['gene']:12s} [{r['labels']}]  "
          f"r_pfc={r['r_pfc']:+.2f} r_hip={r['r_hip']:+.2f} r_hypo={r['r_hypo']:+.2f}")

# ── 2. load pseudobulks (once, shared across all resolutions) ─────────────────
print("Loading PFC pseudobulk…")
with open("~/atac_processing_techniques/cache/pfc_peak_pseudobulk.pkl", "rb") as f:
    pfc_all = pickle.load(f)
pfc_ages       = pfc_all["age"].astype(np.float32)
pfc_all_sum    = pfc_all["peak_sum"].astype(np.float32)
pfc_peak_names = np.asarray(pfc_all["peak_names"])

print("Loading hippocampus pseudobulk…")
with open("~/pfc_to_hippocampus/cache/hip_peak_pseudobulks_in_pfc_space.pkl", "rb") as f:
    hip = pickle.load(f)
hip_ages = hip["age"].astype(np.float32)
hip_sum  = hip["all_cells"]["peak_sum"].astype(np.float32)
hip_cc   = hip["all_cells"]["cell_counts"].astype(np.float32)

print("Loading mouse hypothalamus pseudobulk + liftover…")
with open(ROOT.parent / "pfc_to_hypo" / "cache" / "hypo_pseudobulks.pkl", "rb") as f:
    hypo = pickle.load(f)
with open(ROOT.parent / "pfc_to_hypo" / "cache" / "mouse_peak_liftover.pkl", "rb") as f:
    liftover = pickle.load(f)
hypo_age_letters = np.array(hypo["age_letters"])
hypo_sum = hypo["all_cells"]["sum"].astype(np.float32)
hypo_cc  = hypo["all_cells"]["cc"].astype(np.float32)

def cpm_log1p(X):
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return np.log1p(X / rs * 1e6)

print("Normalising…")
pfc_norm  = cpm_log1p(pfc_all_sum)
hip_norm  = cpm_log1p(hip_sum / np.clip(hip_cc[:, None], 1, None))
hypo_norm = cpm_log1p(hypo_sum / np.clip(hypo_cc[:, None], 1, None))

# ── age-tertile buckets per cohort ─────────────────────────────────────────────
pfc_bucket = pd.qcut(pfc_ages, 3, labels=["Y", "M", "A"]).to_numpy(dtype=str)
hip_bucket = pd.qcut(hip_ages, 3, labels=["Y", "M", "A"]).to_numpy(dtype=str)
hypo_bucket = hypo_age_letters   # already native Young/Middle/Aged groups

pfc_bucket_mask  = {b: pfc_bucket == b for b in "YMA"}
hip_bucket_mask  = {b: hip_bucket == b for b in "YMA"}
hypo_bucket_mask = {b: hypo_bucket == b for b in "YMA"}

# ── 3. vectorised interval indices ─────────────────────────────────────────────
print("Indexing PFC peak intervals…")
split = pd.Series(pfc_peak_names).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
pfc_chrom = split["chrom"].to_numpy()
pfc_start = split["start"].to_numpy(dtype=np.int64)
pfc_end   = split["end"].to_numpy(dtype=np.int64)

print("Indexing mouse liftover positions…")
mouse_chrom = np.array([lo[0] if lo is not None else "" for lo in liftover])
mouse_pos   = np.array([lo[1] if lo is not None else -1 for lo in liftover], dtype=np.int64)

ens = pyensembl.EnsemblRelease(77)

def nearest_gene_full(chrom, mid, search_radius=3_000_000):
    """Single closest protein-coding gene to `mid` (0 distance if `mid`
    falls inside the gene body), with its longest-transcript exon structure
    for drawing."""
    contig = chrom.replace("chr", "")
    genes = [g for g in ens.genes_at_locus(contig, max(0, mid - search_radius), mid + search_radius)
              if g.biotype == "protein_coding"]
    if not genes:
        return None
    def dist(g):
        return 0 if g.start <= mid <= g.end else min(abs(g.start - mid), abs(g.end - mid))
    g = min(genes, key=dist)
    tids = ens.transcript_ids_of_gene_id(g.gene_id)
    best = max(tids, key=lambda t: ens.transcript_by_id(t).end - ens.transcript_by_id(t).start)
    tx = ens.transcript_by_id(best)
    return dict(name=g.gene_name, start=g.start, end=g.end, strand=g.strand,
                exons=tx.exon_intervals, distance=dist(g))

def r_label(r, p):
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"r={r:.2f}{stars}"

def min_draw_width(window_span):
    return max(300, window_span * 0.002)

def bucket_zscores(norm_mat, col_idx, bucket_mask):
    """Per-peak age-bucket means expressed as z-scores against that peak's
    own cross-donor mean/SD (computed over ALL donors in the cohort)."""
    sub = norm_mat[:, col_idx]
    mean = sub.mean(axis=0)
    std  = sub.std(axis=0)
    eps  = 1e-3
    return {b: (sub[m].mean(axis=0) - mean) / (std + eps) for b, m in bucket_mask.items()}

def draw_gene_track(ax, lay, lo, hi, span, r_text=None):
    n_rows = lay["n_rows"]
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, n_rows)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if r_text is not None:
        # short enough ("r=-0.48***") to fit as an axis title at this column
        # width, unlike a full "Hippocampus r=-0.42**" cohort+r combo
        ax.set_title(r_text, fontsize=5.6, pad=2, color="#444444")
    if lay["placed"]:
        for g, gr in lay["placed"]:
            y = n_rows - gr - 0.5
            gs_, ge_ = max(g["start"], lo), min(g["end"], hi)
            ax.plot([gs_, ge_], [y, y], color=GENE_C, lw=1.0, zorder=2)
            # strand-direction chevrons along the intron line (UCSC/IGV-style
            # gene-model convention), spaced by a fixed fraction of the window
            # so gene width alone doesn't overcrowd the arrows
            n_chevron = int(np.clip((ge_ - gs_) / span * 40, 2, 25))
            chevron_xs = np.linspace(gs_, ge_, n_chevron + 2)[1:-1]
            chevron_marker = ">" if g["strand"] == "+" else "<"
            ax.plot(chevron_xs, [y] * len(chevron_xs), marker=chevron_marker,
                    markersize=3.2, color=GENE_C, linestyle="none", zorder=1,
                    clip_on=True)
            for (es, ee) in g["exons"]:
                es_, ee_ = max(es, lo), min(ee, hi)
                if ee_ <= es_:
                    continue
                ax.add_patch(Rectangle((es_, y - 0.09), ee_ - es_, 0.18,
                                        facecolor=GENE_C, edgecolor="none", zorder=3))
            arrow = "→" if g["strand"] == "+" else "←"
            lbl_x = min(max((gs_ + ge_) / 2, lo + span * 0.01), hi - span * 0.01)
            ax.text(lbl_x, y + 0.22, f"{arrow} {g['name']}", ha="center", va="bottom",
                     fontsize=6.5, style="italic", color=GENE_C, clip_on=True)
    elif lay["nearest"] is not None:
        # short + clipped: at this column width (FIG_W/3) the full sentence
        # used in the wide multi-gene version overflows the axis and — since
        # matplotlib text isn't clipped by default — inflates the whole
        # figure's bbox_inches="tight" canvas, visually shrinking everything
        # else. Only happens when the nearest gene falls outside this
        # particular half-width (e.g. NFE2L3 at 150/300kb, gene 297kb away).
        nb = lay["nearest"]
        x = hi - span * 0.02 if nb["side"] == "right" else lo + span * 0.02
        ha = "right" if nb["side"] == "right" else "left"
        arrow = "→" if nb["side"] == "right" else "←"
        ax.text(x, 0.5, f"{arrow} {nb['name']} ({nb['distance']/1000:.0f} kb)",
                 ha=ha, va="center", fontsize=5.5, style="italic", color="#888888",
                 clip_on=True)
    else:
        ax.text(0.5, 0.5, "no gene nearby", ha="center", va="center",
                 fontsize=5.5, color="#999999", transform=ax.transAxes, clip_on=True)
    ax.set_xticks([])

def draw_peaks_track(ax, lo, hi, positions, target_pos, sign_c, dw, count_label):
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for (s_, e_) in positions:
        c_ = (s_ + e_) / 2
        w_ = max(e_ - s_, dw)
        is_target = target_pos is not None and s_ == target_pos[0] and e_ == target_pos[1]
        ax.add_patch(Rectangle((c_ - w_ / 2, 0.15), w_, 0.7,
                                facecolor=sign_c if is_target else "#bbbbbb",
                                edgecolor="black" if is_target else "none",
                                linewidth=0.8 if is_target else 0, zorder=3 if is_target else 2))
    ax.set_xticks([])
    ax.text(1.0, 0.5, count_label, transform=ax.transAxes,
             ha="right", va="center", fontsize=5.8, color="#666666", clip_on=True)

COHORTS = ["PFC", "Hippocampus", "Mouse hypothalamus"]
COHORTS_SHORT = ["PFC", "Hipp.", "M. hypo."]
GENE_ROW_H = 0.42
GENE_BASE_H = 0.5
PEAKS_H  = 0.55
SIG_H    = 1.1
TITLE_H  = 0.4
HEADER_H = 0.35
GROUP_GAP = 0.9
FIG_W = 9.5 / 3   # same narrow width for every resolution

def build_figure(half_width, label):
    print(f"\n=== half-width = {half_width:,} bp ({label}) ===")

    # ── per-region window + nearest-gene lookup (pre-pass, sizes the GridSpec) ─
    layouts = []
    for ri, reg in regions.iterrows():
        chrom, rest = reg["peak_name"].split(":")
        pk_start, pk_end = (int(x) for x in rest.split("-"))
        pk_mid = (pk_start + pk_end) // 2
        gene   = reg["gene"]
        lo, hi = pk_mid - half_width, pk_mid + half_width
        nearest = nearest_gene_full(chrom, pk_mid)
        if nearest is not None and nearest["start"] < hi and nearest["end"] > lo:
            placed, n_rows = [(nearest, 0)], 1
            offpanel = None
        else:
            placed, n_rows = [], 1
            offpanel = None
            if nearest is not None:
                side = "left" if (nearest["start"] + nearest["end"]) / 2 < lo else "right"
                offpanel = dict(name=nearest["name"], distance=nearest["distance"], side=side)
        dist_str = f"{nearest['distance']/1000:.0f} kb away" if nearest else "no nearby gene"
        print(f"  {gene:12s} nearest gene = {nearest['name'] if nearest else 'none'} ({dist_str})")
        layouts.append(dict(chrom=chrom, pk_start=pk_start, pk_end=pk_end, pk_mid=pk_mid,
                             gene=gene, lo=lo, hi=hi, span=hi - lo, placed=placed,
                             n_rows=n_rows, nearest=offpanel, reg=reg))

    heights = [HEADER_H]
    for lay in layouts:
        heights.append(TITLE_H)
        heights.append((GENE_BASE_H + GENE_ROW_H * lay["n_rows"]) + PEAKS_H + 3 * SIG_H)
        heights.append(GROUP_GAP)
    heights = heights[:-1]

    fig = plt.figure(figsize=(FIG_W, sum(heights) * 0.62 + 1.2))
    gs = GridSpec(len(heights), 1, height_ratios=heights, hspace=0.0)

    header = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.16)
    for col, cname in enumerate(COHORTS_SHORT):
        axh = fig.add_subplot(header[0, col])
        axh.axis("off")
        axh.text(0.5, 0.5, cname, ha="center", va="center", fontsize=6.5, fontweight="bold")

    row = 1   # row 0 is the cohort-name header
    for lay in layouts:
        reg, chrom = lay["reg"], lay["chrom"]
        pk_start, pk_end, gene = lay["pk_start"], lay["pk_end"], lay["gene"]
        lo, hi, span = lay["lo"], lay["hi"], lay["span"]
        dw     = min_draw_width(span)
        sign_c = UP_C if reg["r_pfc"] > 0 else DOWN_C

        ax_title = fig.add_subplot(gs[row])
        ax_title.axis("off")
        grid_slot = gs[row + 1]
        row += 3  # title + grid + gap

        title = (f"{gene}  —  {chrom}:{pk_start}-{pk_end}   [{reg['labels']}]     "
                 f"PFC r={reg['r_pfc']:+.2f}   Hip r={reg['r_hip']:+.2f}   "
                 f"MouseHypo r={reg['r_hypo']:+.2f}")
        ax_title.set_title(title, fontsize=6.6, fontweight="bold", loc="left", pad=0, color=sign_c)

        win_mask = (pfc_chrom == chrom) & (pfc_start < hi) & (pfc_end > lo)
        win_idx  = np.nonzero(win_mask)[0]
        target_i = [i for i in win_idx if int(pfc_start[i]) == pk_start]
        target_i = target_i[0] if target_i else None

        mwin_mask = (mouse_chrom == chrom) & (mouse_pos >= lo) & (mouse_pos <= hi)
        mwin_idx  = np.nonzero(mwin_mask)[0]
        m_pos_list = [(int(mouse_pos[i]) - 250, int(mouse_pos[i]) + 250) for i in mwin_idx]
        target_mouse_idx = [i for i in mwin_idx if pk_start <= int(mouse_pos[i]) <= pk_end]
        target_m_pos = (int(mouse_pos[target_mouse_idx[0]]) - 250,
                         int(mouse_pos[target_mouse_idx[0]]) + 250) if target_mouse_idx else None

        pfc_pos = [(int(pfc_start[i]), int(pfc_end[i])) for i in win_idx]
        pfc_bucket_vals = bucket_zscores(pfc_norm, win_idx, pfc_bucket_mask)
        r_pfc_local = p_pfc_local = None
        if target_i is not None:
            r_pfc_local, p_pfc_local = pearsonr(pfc_ages, pfc_norm[:, target_i])

        hip_bucket_vals = bucket_zscores(hip_norm, win_idx, hip_bucket_mask)
        r_hip_local = p_hip_local = None
        if target_i is not None:
            r_hip_local, p_hip_local = pearsonr(hip_ages, hip_norm[:, target_i])

        hypo_bucket_vals = (bucket_zscores(hypo_norm, mwin_idx, hypo_bucket_mask)
                             if len(mwin_idx) else {b: np.array([]) for b in "YMA"})
        r_hypo_local = p_hypo_local = None
        if target_mouse_idx:
            hypo_target_expr = hypo_norm[:, target_mouse_idx].mean(axis=1)
            hypo_age_num = np.where(hypo_age_letters == "Y", 1, np.where(hypo_age_letters == "M", 2, 3))
            r_hypo_local, p_hypo_local = pearsonr(hypo_age_num, hypo_target_expr)

        cohort_data = [
            (pfc_pos,  pfc_bucket_vals,  (pk_start, pk_end), r_pfc_local,  p_pfc_local,
             pfc_pos, f"{len(win_idx)} peaks"),
            (pfc_pos,  hip_bucket_vals,  (pk_start, pk_end), r_hip_local,  p_hip_local,
             pfc_pos, f"{len(win_idx)} peaks"),
            (m_pos_list, hypo_bucket_vals, target_m_pos,     r_hypo_local, p_hypo_local,
             m_pos_list, f"{len(mwin_idx)} peaks"),
        ]

        n_rows_gene = lay["n_rows"]
        gene_h = GENE_BASE_H + GENE_ROW_H * n_rows_gene
        inner = GridSpecFromSubplotSpec(5, 3, subplot_spec=grid_slot,
                                         height_ratios=[gene_h, PEAKS_H, SIG_H, SIG_H, SIG_H],
                                         hspace=0.12, wspace=0.16)
        for col, (cname, (positions, bucket_vals, target_pos, r, p, pk_positions, pk_label)) \
                in enumerate(zip(COHORTS, cohort_data)):
            ax_g = fig.add_subplot(inner[0, col])
            draw_gene_track(ax_g, lay, lo, hi, span, r_text=r_label(r, p) if r is not None else None)

            ax_p = fig.add_subplot(inner[1, col])
            draw_peaks_track(ax_p, lo, hi, pk_positions, target_pos, sign_c, dw, pk_label)

            vmax = max((np.abs(v).max() if v.size else 0.0) for v in bucket_vals.values())
            vmax = vmax * 1.3 if vmax > 0 else 1.0
            for rowi, b in enumerate("YMA"):
                ax = fig.add_subplot(inner[2 + rowi, col])
                ax.set_xlim(lo, hi)
                ax.set_ylim(-vmax, vmax)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                ax.axhline(0, color="#999999", lw=0.6, zorder=1)
                vals = bucket_vals[b]
                for (s_, e_), v in zip(positions, vals):
                    c_ = (s_ + e_) / 2
                    w_ = max(e_ - s_, dw)
                    is_target = target_pos is not None and s_ == target_pos[0] and e_ == target_pos[1]
                    ax.add_patch(Rectangle((c_ - w_ / 2, 0), w_, v, facecolor=AGE_COLORS[b],
                                            edgecolor="black" if is_target else "none",
                                            linewidth=1.1 if is_target else 0,
                                            alpha=0.95 if is_target else 0.8,
                                            zorder=3 if is_target else 2))
                ax.set_xticks([])
                if not vals.size:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=5,
                             color="#999999", transform=ax.transAxes)
                if col == 0:
                    ax.set_ylabel(AGE_ROW_LABELS[b], fontsize=6, rotation=0, ha="right", va="center",
                                   color=AGE_COLORS[b], fontweight="bold", labelpad=4)
                    ax.tick_params(labelsize=5)
                else:
                    ax.set_yticks([])
                if rowi == 2:
                    xticks = np.linspace(lo, hi, 3)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f"{x/1e6:.2f}" for x in xticks], fontsize=4.8)
                    ax.set_xlabel(f"{chrom} (Mb)", fontsize=5.4)

    handles = [Rectangle((0, 0), 1, 1, facecolor=AGE_COLORS[b], label=f"{AGE_ROW_LABELS[b]}")
               for b in "YMA"]
    fig.legend(handles=handles, loc="upper right", fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.995, 0.998), title="Age bucket", title_fontsize=6.5)

    fig.suptitle("SHAP top-10 peaks, all-3-cohort detected + concordant sign — "
                 f"+/-{label} window, nearest-gene-only\n(rows: young→old, columns: cohort; "
                 "bar height = per-peak z-score vs. own cross-donor mean/SD)",
                 fontsize=7.5, fontweight="bold", x=0.02, ha="left", y=0.999)
    fig.tight_layout(rect=[0.01, 0.0, 1, 0.97])

    for ext in ("pdf", "png"):
        p = FIG / f"shap_top10_cross_dataset_tracks_by_age_rect_{label}_nearestgene.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved {p}")
    plt.close(fig)

build_figure(HALF_WIDTH, WIDTH_LABEL)

print("\nDone.")
