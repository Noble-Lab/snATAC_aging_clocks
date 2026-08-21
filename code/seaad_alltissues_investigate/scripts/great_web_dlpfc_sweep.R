# Submit the DLPFC SHAP / SHAP-diff-pct top-100 peak sets (per cell type /
# all cells) to the REAL GREAT web server (great.stanford.edu) via
# rGREAT::submitGreatJob(), and retrieve the "MSigDB Pathway" ontology table
# (contains REACTOME_*, KEGG_*, BIOCARTA_*, PID_* gene sets).
#
# Mirrors ~/pfc_internal_valid/scripts/great_web_sweep.R. The current live
# GREAT deployment defaults to v4.0.4 for hg38, which dropped all pathway
# ontologies (only GO/Phenotype/Genes remain). Pathway ontologies are still
# served under the older v3.0.0 engine, which only supports hg19/mm9/mm10 --
# so peaks are lifted hg38->hg19 first.
#
# Loads: results/great_web_dlpfc_beds/{label}__{tag}.bed
# Saves: results/great_web_dlpfc_raw/{label}__{tag}.csv  (one per job, MSigDB Pathway table)
#
# Usage:
#   env PATH="~/micromamba/envs/renv/bin:$PATH" \
#     ~/micromamba/envs/renv/bin/Rscript scripts/great_web_dlpfc_sweep.R

suppressMessages(library(rGREAT))
suppressMessages(library(GenomicRanges))
suppressMessages(library(rtracklayer))

ROOT    <- "~/seaad_alltissues_investigate"
RESULTS <- file.path(ROOT, "results")
BED_DIR <- file.path(RESULTS, "great_web_dlpfc_beds")
OUT_DIR <- file.path(RESULTS, "great_web_dlpfc_raw")
CHAIN   <- file.path(RESULTS, "hg38ToHg19.over.chain")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

chain <- import.chain(CHAIN)

read_bed_as_gr <- function(path) {
  df <- read.table(path, sep = "\t", col.names = c("chr", "start", "end"))
  GRanges(seqnames = df$chr, ranges = IRanges(start = df$start + 1, end = df$end))
}

liftover_hg38_to_hg19 <- function(gr_hg38) {
  lst <- liftOver(gr_hg38, chain)
  gr <- unlist(lst)
  gr[!duplicated(gr)]
}

bed_files <- sort(list.files(BED_DIR, pattern = "\\.bed$", full.names = TRUE))
cat(sprintf("Found %d BED files in %s\n", length(bed_files), BED_DIR))

for (bed_path in bed_files) {
  tag <- tools::file_path_sans_ext(basename(bed_path))
  out_csv <- file.path(OUT_DIR, sprintf("%s.csv", tag))
  if (file.exists(out_csv)) {
    cat(sprintf("[skip] %s already done\n", tag))
    next
  }

  cat(sprintf("\n=== %s ===\n", tag))
  gr_hg38 <- read_bed_as_gr(bed_path)
  gr_hg19 <- liftover_hg38_to_hg19(gr_hg38)
  cat(sprintf("  %d/%d peaks lifted hg38->hg19\n", length(gr_hg19), length(gr_hg38)))

  job <- tryCatch(
    submitGreatJob(
      gr_hg19, genome = "hg19", version = "3.0.0", rule = "basalPlusExt",
      adv_upstream = 5, adv_downstream = 1, adv_span = 1000,
      request_interval = 20, max_tries = 20, verbose = FALSE
    ),
    error = function(e) { cat("  SUBMIT ERROR:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(job)) next

  tbl <- tryCatch(
    getEnrichmentTables(job, ontology = "MSigDB Pathway",
                        request_interval = 20, max_tries = 20, verbose = FALSE),
    error = function(e) { cat("  FETCH ERROR:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(tbl)) next
  if (is.list(tbl) && !is.data.frame(tbl)) tbl <- tbl[[1]]

  tbl$tag <- tag
  tbl$n_regions <- length(gr_hg38)
  tbl$n_regions_lifted <- length(gr_hg19)
  write.csv(tbl, out_csv, row.names = FALSE)
  cat(sprintf("  Saved %s (%d pathway terms)\n", out_csv, nrow(tbl)))

  Sys.sleep(5)  # be polite to the shared GREAT server between jobs
}

cat("\nDone.\n")
