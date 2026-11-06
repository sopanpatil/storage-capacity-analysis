#!/usr/bin/env bash
#
# download_caravan.sh — fetch the full Caravan v1.6 and GRDC-Caravan v0.6 netCDF archives.
#
# Download into a temporary folder, then unpack with extract_caravan.sh. If Zenodo answers
# HTTP 403 from your host, download the two files elsewhere and copy them into TMP_DIR.
#
#     bash download_caravan.sh TMP_DIR
#
# Needs ~33 GB free in TMP_DIR. Resumable (wget -c) and md5-checked; safe to re-run.

set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 TMP_DIR" >&2; exit 1; }
DEST="$1"

# filename | url | md5 (from the Zenodo record pages)
FILES=(
  "Caravan-nc.tar.gz|https://zenodo.org/records/15529786/files/Caravan-nc.tar.gz?download=1|ad3a7767db059f218380e9cce5f72452"
  "GRDC_Caravan_extension_nc.zip|https://zenodo.org/records/15349031/files/GRDC_Caravan_extension_nc.zip?download=1|c88a29fe7c90b5248a03d35d92b2815d"
)

mkdir -p "$DEST" && [[ -w "$DEST" ]] || { echo "FAIL: cannot write to '$DEST' on $(hostname)" >&2; exit 2; }
echo "Destination: $DEST"

code=$(curl -s -o /dev/null -w '%{http_code}' -I -L --max-time 30 --user-agent 'Mozilla/5.0' \
       'https://zenodo.org/records/15349031' || echo 000)
[[ "$code" == 200 ]] || { echo "FAIL: Zenodo returned HTTP $code from $(hostname)." >&2; exit 3; }

cd "$DEST"
for entry in "${FILES[@]}"; do
  IFS='|' read -r name url md5 <<< "$entry"
  echo "=== $name ==="
  wget -c --tries=5 --timeout=60 --user-agent='Mozilla/5.0' -O "$name" "$url"
  echo "$md5  $name" | md5sum -c - || { echo "CHECKSUM FAILED: $name" >&2; exit 1; }
done
ls -lh "$DEST"
echo "Done. Next: bash extract_caravan.sh $DEST RAW_DIR"
