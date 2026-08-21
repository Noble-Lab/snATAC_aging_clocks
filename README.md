# Reproduction Instructions

## Layout

```
code/
  _shared/                       stats_log.py / merge_outputs.py / harvest_seaad_ridge_coefs.py (§5)
  environment/                   conda env spec + activation hook (§1)
  pfc_internal_valid/            scripts/, motif_databases/, reference/
  pfc_to_hippocampus/            scripts (flat)
  pfc_to_hypo/
  pfc_to_retinal/
  pfc_to_zebrafish_retinal/
  seaad_alltissues_investigate/  scripts/, data/metadata/, cache/ (pre-built donor manifest)
  atac_processing_techniques/    scripts/, _external_deps/
  pfc_to_perturbation/           scripts (flat), data/ucsc_cache/, reference/
docs/
  REPRODUCIBILITY_REPORT.md
  DATA_FLOW_DOCUMENTATION.md
outputs/
  statistical_tests/              one CSV per (project, analysis) (generated, §5)
  clock_coefficients/              one CSV per (project, clock_name) (generated, §5)
figures/                         the 24 target figures, one subfolder per project (generated)
```

**Paths.** Every script locates its own project directory as `Path(__file__).resolve().parent`
(or `.parent.parent` for the 3 projects with a `scripts/` subdir), so cache/, results/, figures/,
and logs/ are created next to the script itself -- this works whether you run scripts in place
inside this bundle (`code/<project>/...`) or copy a project's files elsewhere, as long as sibling
projects it cross-references (e.g. `pfc_to_hypo`) stay one level up from it, matching this
bundle's own layout. The only remaining **hardcoded absolute paths** are (a) very large raw source datasets (§2) and (b) a handful of this project's large intermediate caches (documented at the top of every script that uses one, e.g.
`pfc_to_hippocampus`'s `cache/pfc_per_ct_peak_pseudobulks.pkl`, 579MB) -- these stay at their
original `~/<project>/...` locations rather than being duplicated into this bundle.
To run the scripts:

1. Ensure the datasets in §2 are present at the paths listed (and the large intermediate
   caches noted above, if not regenerating them from scratch).
2. Build/activate the `reprod` conda environment (§1).
3. Run the commands in §3, in order, from inside this bundle's `code/<project>/` directories (or
   copy a project's files to `~/<project>/` and run there, as in the original layout -- both work).
4. Run `code/_shared/merge_outputs.py` (§5) to build the per-analysis summary tables, and copy
   each project's figure(s) into this repo's `figures/<project>/`.

## 1. Reproduction environment

```bash
micromamba create -y -p /path/to/envs/reprod -f code/environment/reprod_env.yml
# code/environment/reprod_env_full_export.yml is the exact fully-pinned export, for
# exact-version reproduction if the loose spec above resolves differently later.

# The env's own lib/ ships a newer libstdc++ than the system one (needed by
# scipy/matplotlib/xgboost/shap). Either install the persistent hook:
mkdir -p /path/to/envs/reprod/etc/conda/{activate.d,deactivate.d}
cp code/environment/activate.d/libstdcxx_path.sh            /path/to/envs/reprod/etc/conda/activate.d/
cp code/environment/activate.d/deactivate_libstdcxx_path.sh /path/to/envs/reprod/etc/conda/deactivate.d/libstdcxx_path.sh
# ...or, for a single shell session, export it directly:
export LD_LIBRARY_PATH=/path/to/envs/reprod/lib:$LD_LIBRARY_PATH

# numpy gets pulled to 2.x by pyensembl's transitive dependencies; pin it back:
conda activate reprod
pip install "numpy==1.26.4" --no-deps --force-reinstall

source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate reprod
```

The MEME suite (`ame`, `meme`, `fasta-get-markov`, `tomtom`, `fimo`, v5.5.9) and R (`Rscript`,
with `rGREAT`, `GenomicRanges`, `rtracklayer`) are both installed inside this same environment.

## 2. Data required prior to replication

| Path | Size | Source |
|---|---|---|
| `~/data_back/PFC_brain_multiome/{final_atac_data,final_rna_data}.h5ad` | 14GB + 18GB | Zenodo 18394349 |
| `~/data_back/GSE278576_hippocampus/{raw,processed}` | 4.6GB + 1.2GB | GEO GSE278576 |
| `~/hypo_atac.h5ad` | 5.4GB | pre-processed mouse hypothalamus ATAC |
| `~/pfc_to_retinal/GSE325478_raw.tar` (or extracted `multiome_raw/*.h5`) | 2.0GB | GEO GSE325478 |
| `~/pfc_to_zebrafish_retinal/GSE325620_RAW.tar` | 1.5GB | GEO GSE325620 |
| `~/data/parkinson_igvf/` | ~104GB | Parkinson's source dataset |
| `~/pfc_to_perturbation/data/GSE294103_atac_seq_normalized_reads.csv.gz` | 18.6MB | GEO GSE294103 |
| `~/pfc_to_mouse/cache/{pfc_X_tile50kb.pkl, feat_idx_tile50kb.pkl}` | 126MB | PFC to mouse tiles |
| `~/atac_processing_techniques/cache/{pfc_peak_pseudobulk.pkl,seaad_peak_pseudobulk.pkl}` | 280MB | Healthy PFC and SEA-AD DLPFC pre-processed pseudobulks (218k overlapping peaks) |
| `~/age_accel_per_cell_type/cache/{seaad_pseudobulks_by_ct.pkl, v3/pfc_pseudobulks_by_ct_v3.pkl}` | 73MB + 2.25GB |  cell-type specific Healthy PFC and SEA-AD DLPFC pre-processed pseudobulks |
| `~/reference_data/dnam_clocks/{cortical_clock,mouse_clock,zebrafish_clock}/` | 145 MB | Meer and Shireby DNAm-clock CpG coordinates |


## 3. Per-subfigure run order

Activate `reprod` first (§1) for every command. `cd` into the relevant project directory before
running.

### 3.1 pfc_internal_valid (subfigures 1–4, 9–11)

```bash
cd ~/pfc_internal_valid
python scripts/shap_enrichment_crossdataset.py            # -> figures/shap_barplot.pdf (subfig 4)
python scripts/great_web_export_beds.py
Rscript scripts/great_web_sweep.R                          # live GREAT web submission
python scripts/great_web_combine.py
python scripts/great_web_sig.py                            # -> figures/pathway_enrichment_GREATweb_sig_top2000.pdf (subfig 1)
python scripts/meme_chip_sweep.py                          # writes AME's input FASTAs
python scripts/ame_max_scoring_sweep.py                    # runs AME (needs motif_databases/)
python scripts/ame_max_scoring_plot.py                      # -> figures/ame_max_scoring/ame_max_spearman_combined_top100.pdf (subfig 2)
python scripts/allpeaks_zscore_cv.py                         # -> figures/allpeaks_5foldCV.pdf (subfig 3)
python scripts/shap_clock_specific_spearman.py                # -> figures/shap_clock_specific_spearman_combined.pdf (subfig 9)
                                                                # (needs pfc_to_hippocampus/pfc_to_hypo/pfc_to_retinal/
                                                                #  pfc_to_zebrafish_retinal caches — run §3.2-3.5 first)
python scripts/shap_top10_cross_dataset_tracks_by_age_rect_nearestgene.py
                                                                # -> figures/shap_top10_cross_dataset_tracks_by_age_rect_150kb_nearestgene.pdf (subfig 10)
python scripts/cortical_clock_enrichment.py
python scripts/plot_cortical_clock_enrichment_v4.py             # -> figures/cortical_clock_enrichment_v4.pdf (subfig 11)
```

### 3.2 pfc_to_hippocampus (subfigures 5, 13)

```bash
cd ~/pfc_to_hippocampus
python pfc_to_hip_atac_clock.py           # builds cache/pfc_peaks_to_hip_tiles_M.pkl, cache/pfc_per_ct_tile_pseudobulks.pkl
python plot_scatter_tile_zscore_horiz_nr2f2inh.py   # builds cache/hip_pseudobulks_nr2f2inh.pkl -> figures/scatter_tile_zscore_horiz_NR2F2inh.pdf (subfig 5)
python pfc_to_hip_atac_clock_peaks.py     # builds cache/hip_peak_pseudobulks_in_pfc_space.pkl, cache/pfc_per_ct_peak_pseudobulks.pkl
python compute_peak_importance.py         # -> results/peak_importance.pkl; logs Ridge coefficients (§5)
python scripts_cortical_clock_enrichment.py
python scripts_plot_cortical_clock_enrichment_v4.py  # -> figures/cortical_clock_enrichment_v4.pdf (subfig 13)
```

### 3.3 pfc_to_hypo (subfigures 6, 12)

```bash
cd ~/pfc_to_hypo
python pfc_to_hypo_atac_clock.py     # builds cache/hypo_pseudobulks.pkl
python pfc_to_hypo_glu_gaba.py       # builds cache/hypo_glu_gaba_pseudobulks.pkl
python pfc_to_hypo_age_map.py        # builds cache/intersect_all_map.pkl, cache/mouse_peak_liftover.pkl
python plot_violin_v2.py             # -> figures/violin_intersect_all_v2.pdf (subfig 6)
python compute_peak_importance.py    # -> results/peak_importance.pkl; logs Ridge coefficients (§5)
python scripts_mouse_clock_enrichment.py
python scripts_plot_mouse_clock_enrichment_v4.py   # -> figures/mouse_clock_enrichment_v4.pdf  
```

### 3.4 pfc_to_retinal  

```bash
cd ~/pfc_to_retinal
# tar -xf GSE325478_raw.tar -C multiome_raw/
python multiome_celltypes.py
python multiome_allcell_clock_per_ct.py        # builds cache/multiome_pseudobulks.pkl -> figures/multiome/analysis_allcell_clock_per_ct_scatter.pdf 
```

### 3.5 pfc_to_zebrafish_retinal  

```bash
cd ~/pfc_to_zebrafish_retinal
python zebrafish_multiome_celltypes.py    # streams samples from GSE325620_RAW.tar
python zebrafish_allcell_clock_per_ct.py  # -> figures/multiome_clocks_final/scatter_allcellclock_percelltype_atac.pdf  
```

### 3.6 seaad_alltissues_investigate 

```bash
cd ~/seaad_alltissues_investigate
python scripts/xds_pfc357_peaks_to_dlpfc.py       # -> results/xds_pfc357_peaks_{label}_all.csv x7  
python scripts/shap_sex_high_adnc.py               # -> cache/shap_sex_pct_cache.pkl (feeds fig 19)
python scripts/lm2factor_peak_contribution.py       # -> results/lm_contrib_peak_rankings_{label}.csv x7, cache/ridge_coef_cache.pkl 
python scripts/shap_dlpfc_analysis.py                # -> cache/shap_adnc_cache.pkl (feeds figs 14, 15)
python scripts/great_web_shaptop_export_beds.py
python scripts/great_web_shapdiffpct_export_beds.py
Rscript scripts/great_web_dlpfc_sweep.R                # live GREAT web submission
python scripts/great_web_dlpfc_combine.py
python scripts/great_web_dlpfc_top5_plot.py             # -> figures/dlpfc_pathway_reactome_GREATweb_top5_{shaptop2000,shapdiffpct_top5_top2000}.pdf (subfig 14, 15)
python scripts/plot_adnc_residuals.py                    # -> figures/dlpfc_xds_pfc357_peaks_{7 cell types}_all_adnc_violin.pdf (subfig 16, x7 files)
python scripts/plot_adnc_violin_sexsplit_twopanel.py       # -> figures/dlpfc_xds_pfc357_peaks_all_adnc_celltype_violin_sexsplit_twopanel.pdf (subfig 18)
python scripts/plot_top_iut_scatter_combined.py              # -> figures/dlpfc_top_iut_posslope_scatter_combined.pdf (subfig 19)
```

### 3.7 atac_processing_techniques 

```bash
cd ~/atac_processing_techniques
Rscript scripts/build_igvf_pseudobulks.R SN <outdir>   # repeat per tissue: CBL, CING, MTG, PUT, SN (SN used for paper analyses)
                                                          # (already-built pseudobulks normally live at
                                                          #  ~/data/parkinson_igvf/pseudobulks/)
Rscript scripts/build_igvf_peak_coords.R      # regenerates cache/igvf_sn_peak_coords.csv (report §3.4)
python scripts/igvf_multitissue_clock.py        # -> figures/igvf_multitissue_violin_{7 cell types}.pdf (subfig 17)
```

### 3.8 pfc_to_perturbation (subfigures 20–24)

```bash
cd ~/pfc_to_perturbation
python pfc_perturbation_clock_v4_peaks.py          # -> cache_v4/gse294103_peak_matrices.pkl, figures_v4/gse294103_sirt6_violin_peaks.pdf (subfig 24)
python pfc_sirt6_shap.py                            # -> cache_v4/gse294103_shap_values.pkl
python pfc_sirt6_shap_pct_dumbbell_labeled_mm10.py    # -> figures_v4/sirt6_shap_diff_top20_pct_dumbbell_labeled_mm10.pdf (subfig 20)
python pfc_sirt6_wt_diff_great_reactome.py             # live GREAT + Enrichr -> figures_v4/sirt6_wt_diff_great_reactome_top100_500_2000.pdf (subfig 21)
python plot_nfkbia_zoom_tracks.py                        # -> figures_v4/nfkbia_peak50298_to_nfkbia.pdf (subfig 22; needs data/ucsc_cache/ucsc_nfkbia_region.json, included)
python plot_nfkbia_peak.py                                # -> figures_v4/nfkbia_peak_violin.pdf (subfig 23)
```
`pfc_sirt6_shap.py` needs `reference/hg38_gene_tss.tsv` (included in this bundle).

## 4. Statistical-tests and clock-coefficients tables

Every script above that runs a statistical test or fits a linear age clock calls
`code/_shared/stats_log.py` (`log_test()` / `log_coef()`), which appends one row per test/
coefficient to `outputs/_raw/<project>_{statistical_tests,clock_coefficients}.csv`. After running
everything in §3:

```bash
python code/_shared/merge_outputs.py
# -> outputs/statistical_tests/<project>__<analysis>.csv       (one file per distinct analysis:
#      test name, groups, n, statistic, p-value, q-value, notes)
# -> outputs/clock_coefficients/<project>__<clock_name>.csv    (one file per distinct clock:
#      modality, cell type, feature, coefficient, rank)
```

merge_outputs.py re-derives the full split from `outputs/_raw/` on every run (clearing and
rewriting `outputs/statistical_tests/` and `outputs/clock_coefficients/`), so it's safe to rerun
after adding more `_raw/` rows -- it also de-duplicates exact-duplicate rows first, which matters
if a script was re-run and its `log_test()`/`log_coef()` calls appended the same rows twice.

`seaad_alltissues_investigate/scripts/lm2factor_peak_contribution.py`'s Ridge fit is cached per
cell type (`cache/ridge_coef_cache.pkl`), so rerunning it against an existing cache skips
retraining and therefore skips `log_coef()`. Run this once to extract those coefficients directly
from the cache instead:

```bash
python code/_shared/harvest_seaad_ridge_coefs.py
```
