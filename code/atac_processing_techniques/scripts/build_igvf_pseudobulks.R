#!/usr/bin/env Rscript
# Build per-tissue pseudobulk matrices from IGVF ATAC RDS files.
# Usage: Rscript build_igvf_pseudobulks.R <tissue> <outdir>
#
# Output:
#   <outdir>/<tissue>_pb.bin   — raw float32 (n_rows × 883711) row-major
#   <outdir>/<tissue>_pb_meta.csv — row_idx, donor_id, ct, age, pd_status

suppressPackageStartupMessages({
  library(SummarizedExperiment)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
tissue  <- args[1]
outdir  <- args[2]

if (is.na(tissue) || is.na(outdir)) {
  stop("Usage: Rscript build_igvf_pseudobulks.R <tissue> <outdir>")
}
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

data_dir  <- "~/data/parkinson_igvf"
N_PEAKS   <- 883711L

# ── CT mapping: IGVF CL_term_name → PFC broad type ───────────────────────────
CT_MAP <- c(
  "glutamatergic neuron"               = "Excitatory",
  "sst GABAergic interneuron"          = "Inhibitory",
  "pvalb GABAergic interneuron"        = "Inhibitory",
  "VIP GABAergic interneuron"          = "Inhibitory",
  "lamp5 GABAergic interneuron"        = "Inhibitory",
  "pvalb chandelier GABAergic interneuron" = "Inhibitory",
  "GABAergic neuron"                   = "Inhibitory",
  "medium spiny neuron"                = "SpinyNeuron",
  "striosomal D1 medium spiny neuron"  = "SpinyNeuron",
  "striosomal D2 medium spiny neuron"  = "SpinyNeuron",
  "D1/D2-hybrid medium spiny neuron"   = "SpinyNeuron",
  "oligodendrocyte"                    = "Oligo",
  "astrocyte"                          = "Astro",
  "microglial cell"                    = "Microglia",
  "oligodendrocyte precursor cell"     = "OPC",
  "substantia nigra dopaminergic neuron" = "DANeuron"
)

# ── Donor metadata ────────────────────────────────────────────────────────────
donor_meta <- read.csv(file.path(data_dir, "donor_metadata.csv"),
                       stringsAsFactors = FALSE)
rownames(donor_meta) <- donor_meta$BannerID

# ── Tissue sample list ────────────────────────────────────────────────────────
tissue_meta <- read.csv(file.path(data_dir, "tissue_metadata.csv"),
                        stringsAsFactors = FALSE)
tissue_df   <- tissue_meta[tissue_meta$BrainRegion == tissue, ]
cat(sprintf("[%s] %d donor-tissue entries\n", tissue, nrow(tissue_df)))

# ── Open binary output immediately — write each row as computed ───────────────
bin_path  <- file.path(outdir, paste0(tissue, "_pb.bin"))
meta_path <- file.path(outdir, paste0(tissue, "_pb_meta.csv"))
con <- file(bin_path, "wb")

meta_rows <- list()
n_rows    <- 0L
t0        <- proc.time()["elapsed"]

write_pb <- function(vec) {
  # Write a plain double vector as 4-byte (float32) values.
  # Using size=4L avoids as.single() and its Csingle attribute,
  # which causes "can only write vector objects" in some R builds.
  writeBin(as.double(vec), con, size = 4L)
}

for (i in seq_len(nrow(tissue_df))) {
  tissue_id <- tissue_df$TissueSampleID[i]
  donor_id  <- tissue_df$BannerDonorID[i]
  atac_acc  <- tissue_df$IGVF_ATAC_ProcessedMatrixFileAccession[i]
  meta_acc  <- tissue_df$IGVF_CellMetadataAccession[i]

  rds_path  <- file.path(data_dir, tissue_id, paste0(atac_acc, ".rds"))
  cell_path <- file.path(data_dir, tissue_id, paste0(meta_acc, ".tsv.gz"))

  if (!file.exists(rds_path)) {
    cat(sprintf("  [%d/%d] SKIP %s — RDS missing\n", i, nrow(tissue_df), tissue_id))
    next
  }
  if (!file.exists(cell_path)) {
    cat(sprintf("  [%d/%d] SKIP %s — TSV missing\n", i, nrow(tissue_df), tissue_id))
    next
  }

  rse <- tryCatch(readRDS(rds_path), error = function(e) NULL)
  if (is.null(rse)) {
    cat(sprintf("  [%d/%d] SKIP %s — readRDS failed\n", i, nrow(tissue_df), tissue_id))
    next
  }
  counts <- assay(rse, 1)   # peaks × cells (dgCMatrix)

  if (nrow(counts) != N_PEAKS) {
    cat(sprintf("  [%d/%d] SKIP %s — unexpected n_peaks %d\n",
                i, nrow(tissue_df), tissue_id, nrow(counts)))
    next
  }

  tsv <- tryCatch(
    read.table(cell_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
  if (is.null(tsv)) {
    cat(sprintf("  [%d/%d] SKIP %s — TSV read failed\n", i, nrow(tissue_df), tissue_id))
    next
  }

  rse_bc <- colnames(counts)
  meta_bc <- tsv$cell_barcode
  shared  <- intersect(rse_bc, meta_bc)
  if (length(shared) == 0) {
    cat(sprintf("  [%d/%d] SKIP %s — barcode mismatch\n", i, nrow(tissue_df), tissue_id))
    next
  }

  counts_sub <- counts[, shared, drop = FALSE]
  ct_sub     <- tsv$CL_term_name[match(shared, meta_bc)]

  dm     <- donor_meta[donor_id, ]
  age    <- if (!is.null(dm) && !is.na(dm$age_at_death)) dm$age_at_death else NA
  pd_val <- if (!is.null(dm) && !is.na(dm$PD))          dm$PD           else NA

  # All-cells pseudobulk — write immediately
  pb_all <- as.numeric(Matrix::rowSums(counts_sub))
  write_pb(pb_all)
  n_rows <- n_rows + 1L
  meta_rows[[n_rows]] <- data.frame(donor_id = donor_id, ct = "All-cells",
                                    age = age, pd_status = pd_val,
                                    stringsAsFactors = FALSE)

  # Per-CT pseudobulks — write immediately
  ct_labels <- CT_MAP[ct_sub]
  for (broad_ct in unique(na.omit(ct_labels))) {
    sel <- which(!is.na(ct_labels) & ct_labels == broad_ct)
    if (length(sel) == 0) next
    pb_ct <- if (length(sel) == 1) as.numeric(counts_sub[, sel])
             else                  as.numeric(Matrix::rowSums(counts_sub[, sel, drop = FALSE]))
    write_pb(pb_ct)
    n_rows <- n_rows + 1L
    meta_rows[[n_rows]] <- data.frame(donor_id = donor_id, ct = broad_ct,
                                      age = age, pd_status = pd_val,
                                      stringsAsFactors = FALSE)
  }

  elapsed <- proc.time()["elapsed"] - t0
  cat(sprintf("  [%d/%d] %s  n_cells=%d  %.1fm elapsed\n",
              i, nrow(tissue_df), tissue_id, length(shared), elapsed / 60))
  rm(rse, counts, counts_sub); gc(verbose = FALSE)
}

close(con)

if (n_rows == 0L) {
  cat("[WARN] No pseudobulks built for", tissue, "\n")
  quit(save = "no")
}

meta_df <- do.call(rbind, meta_rows)
meta_df$row_idx <- seq_len(nrow(meta_df)) - 1L
write.csv(meta_df, meta_path, row.names = FALSE)

cat(sprintf("[%s] Done: %d pseudobulks → %s\n", tissue, n_rows, bin_path))
cat(sprintf("  Matrix: %d rows × %d peaks  (%.0f MB)\n",
            n_rows, N_PEAKS, n_rows * N_PEAKS * 4 / 1e6))
cat(sprintf("  Total elapsed: %.1f min\n", (proc.time()["elapsed"] - t0) / 60))
