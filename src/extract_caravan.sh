#!/usr/bin/env bash
#
# extract_caravan.sh — unpack the archives fetched by download_caravan.sh.
#
#     bash extract_caravan.sh TMP_DIR RAW_DIR
#
# Unpacks into RAW_DIR/caravan_v1.6 and RAW_DIR/grdc_caravan_v0.6. The archives in TMP_DIR are
# left in place; delete them by hand once the extraction looks right (they use ~33 GB).

set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 TMP_DIR RAW_DIR" >&2; exit 1; }
TMP="$1"; RAW="$2"

for f in Caravan-nc.tar.gz GRDC_Caravan_extension_nc.zip; do
  [[ -f "$TMP/$f" ]] || { echo "FAIL: $TMP/$f not found; run download_caravan.sh first" >&2; exit 2; }
done
mkdir -p "$RAW/caravan_v1.6" "$RAW/grdc_caravan_v0.6"

echo "=== Caravan-nc.tar.gz -> $RAW/caravan_v1.6 ==="
tar -xzf "$TMP/Caravan-nc.tar.gz" -C "$RAW/caravan_v1.6"
echo "=== GRDC_Caravan_extension_nc.zip -> $RAW/grdc_caravan_v0.6 ==="
unzip -q -o "$TMP/GRDC_Caravan_extension_nc.zip" -d "$RAW/grdc_caravan_v0.6"

for d in caravan_v1.6 grdc_caravan_v0.6; do
  echo "$d: $(find "$RAW/$d" -name '*.nc' | wc -l) netCDF files, $(du -sh "$RAW/$d" | cut -f1)"
done
echo "If those look right, free the space with: rm $TMP/Caravan-nc.tar.gz $TMP/GRDC_Caravan_extension_nc.zip"
