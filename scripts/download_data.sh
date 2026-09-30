#!/usr/bin/env bash
# Public genome-scale CRISPRi Perturb-seq screens used for training and local validation.
# Resumable: a partial file is continued, a complete one (size == remote) is skipped.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
fetch() {
  local url=$1 out=$2 remote local_size
  remote=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:"{gsub("\r","",$2); n=$2} END{print n}')
  local_size=$( [ -f "$out" ] && stat -f%z "$out" || echo 0 )
  if [ -n "$remote" ] && [ "$local_size" -ge "$remote" ]; then return; fi
  curl -fL --retry 10 --retry-all-errors --retry-delay 5 -C - -o "$out" "$url"
}
FS=https://ndownloader.figshare.com/files
# Replogle et al. 2022 (Cell): K562 essential + RPE1, pseudobulk (raw counts)
fetch "$FS/35773070" replogle_k562_essential_raw_bulk.h5ad
fetch "$FS/35775581" replogle_rpe1_raw_bulk.h5ad
fetch "$FS/35774443" replogle_k562_gwps_raw_bulk.h5ad
# Nadig et al. 2025 (Nat Genet): HepG2 + Jurkat essential screens, single cell (pseudobulked locally)
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl
fetch "$GEO/GSE264667_hepg2_raw_singlecell_01.h5ad" nadig_hepg2_raw_singlecell.h5ad
fetch "$GEO/GSE264667_jurkat_raw_singlecell_01.h5ad" nadig_jurkat_raw_singlecell.h5ad
# Arc Virtual Cell Challenge 2025 (Roohani et al., Cell 2025): H1 hESC CRISPRi, all three
# released splits, single cell, genes indexed by symbol (~34 GB)
VCC=https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025
mkdir -p h1 cd4
fetch "$VCC/train/adata_Training.h5ad" h1/adata_Training.h5ad
fetch "$VCC/validation/adata_Validation.h5ad" h1/adata_Validation.h5ad
fetch "$VCC/test/adata_Test.h5ad" h1/adata_Test.h5ad
# Zhu et al. (Cell 2026, doi:10.1016/j.cell.2026.08.002): genome-scale CRISPRi in primary human CD4+ T cells, publisher
# differential-expression statistics per target and culture condition (~17 GB)
fetch https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad cd4/GWCD4i.DE_stats.h5ad
shasum -a 256 *.h5ad h1/*.h5ad cd4/*.h5ad > SHA256SUMS
