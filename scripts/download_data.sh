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
shasum -a 256 *.h5ad > SHA256SUMS
