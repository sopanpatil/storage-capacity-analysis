#!/usr/bin/env bash
#
# download_ssebop.sh — fetch SSEBop v5 monthly actual ET (USGS FEWS NET, MODIS-based, ~1 km, mm/month)
# for the benchmark period 2003–2020.
#
# Source: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/monthly/etav5/downloads/
# (the sibling monthly/eta/ folder is v4; do not use it).
#
#     bash src/download_ssebop.sh DEST_DIR      # e.g. <benchmarks>/ssebop_v5_monthly
#
# 216 zips, ~140 MB each (~30 GB). Kept zipped: GDAL reads them through /vsizip/. Resumable: a zip is
# skipped when its size matches the server's Content-Length.

set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 DEST_DIR" >&2; exit 1; }
DEST="$1"
BASE=https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/monthly/etav5/downloads
mkdir -p "$DEST"

for year in $(seq 2003 2020); do
  for month in $(seq -w 1 12); do
    f="m${year}${month}.zip"
    want=$(curl -sI -m 60 "$BASE/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:" {print $2}')
    [[ -n "$want" ]] || { echo "no Content-Length for $f" >&2; exit 1; }
    if [[ -f "$DEST/$f" && $(stat -c %s "$DEST/$f") -eq $want ]]; then continue; fi
    curl -s -f -m 1800 -C - -o "$DEST/$f" "$BASE/$f" || curl -s -f -m 1800 -o "$DEST/$f" "$BASE/$f"
    got=$(stat -c %s "$DEST/$f")
    [[ $got -eq $want ]] || { echo "size mismatch $f: $got != $want" >&2; exit 1; }
    echo "$f ok"
  done
done
echo "all 216 months present in $DEST"
