#!/usr/bin/env Rscript
# Build cache/igvf_sn_peak_coords.csv: chrom/start/end coordinates for the
# 883,711-peak IGVF reference used (by scripts/igvf_sn_clock.py and
# scripts/igvf_multitissue_clock.py) to overlap IGVF peaks against PFC peaks.
#
# Every raw per-donor-tissue RDS file under
#   ~/data/parkinson_igvf/*/*.rds
# contains a RangedSummarizedExperiment whose rowRanges() carries the
# identical 883,711-peak set, in the identical row order, for every
# donor/tissue (a shared fixed reference peak set across the whole IGVF PD
# atlas), so any one RDS file is sufficient to extract it from.
#
# rowRanges() also carries an `idx` mcols column, but that column is a
# PER-CHROMOSOME positional index (resets to 1 at the start of each
# chromosome: chr1 idx 1..80871, chr2 idx 1..70525, ...) rather than a global
# ordering key. Chromosomes already appear as contiguous blocks in natural
# row order (chr1, chr2, ..., chr22, chrX), and idx is already monotonic
# non-decreasing within each block, so the peak order used downstream
# (peak_0, peak_1, ...) is the natural row order of rowRanges(), NOT a
# re-sort by `idx` (a global sort by idx alone would corrupt cross-chromosome
# order, since idx repeats 1..N per chromosome).
#
# Usage: Rscript build_igvf_peak_coords.R [rds_path] [out_csv]
#   rds_path defaults to a small-ish representative file (BANN1311_CING).
#   out_csv  defaults to cache/igvf_sn_peak_coords.csv relative to repo root.

suppressPackageStartupMessages({
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
rds_path <- if (length(args) >= 1) args[1] else
  "~/data/parkinson_igvf/BANN1311_CING/IGVFFI0275PPCR.rds"
out_csv  <- if (length(args) >= 2) args[2] else
  "~/atac_processing_techniques/cache/igvf_sn_peak_coords.csv"

N_PEAKS_EXPECTED <- 883711L

cat(sprintf("Reading %s ...\n", rds_path))
rse <- readRDS(rds_path)
rr  <- rowRanges(rse)

if (length(rr) != N_PEAKS_EXPECTED) {
  stop(sprintf("Expected %d peaks, got %d — wrong RDS file?",
               N_PEAKS_EXPECTED, length(rr)))
}

df <- as.data.frame(rr)

# Sanity: idx must be a per-chromosome positional index, monotonic within
# each chromosome block, with chromosome blocks contiguous in row order.
chrom_rle <- rle(as.character(df$seqnames))
if (any(duplicated(chrom_rle$values))) {
  stop("seqnames are not contiguous blocks in row order — assumption violated, aborting")
}
idx_ok <- all(tapply(df$idx, df$seqnames, function(x) !is.unsorted(x)))
if (!idx_ok) stop("idx is not monotonic within a chromosome block — assumption violated")

out <- data.frame(
  peak_name = paste0("peak_", seq_len(nrow(df)) - 1L),
  chrom     = as.character(df$seqnames),
  start     = df$start,
  end       = df$end,
  stringsAsFactors = FALSE
)

# Verify the well-known anchor row before writing.
stopifnot(out$chrom[1] == "chr1", out$start[1] == 804664, out$end[1] == 805164)

write.csv(out, out_csv, row.names = FALSE, quote = TRUE)
cat(sprintf("Wrote %d peaks to %s\n", nrow(out), out_csv))
