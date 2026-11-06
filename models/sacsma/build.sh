#!/usr/bin/env bash
# Compile the NWS SAC-SMA + SNOW-17 Fortran (nwsrfsr 1.0.3, unchanged, in third_party/) into a standalone
# shared library for sacsnow.py. Needs gfortran. No R required at run time.
set -euo pipefail
cd "$(dirname "$0")"
S=../../third_party/nwsrfsr-1.0.3/src; mkdir -p build; cd build; rm -f *.o *.mod
F="-c -fPIC -O2 -ffp-contract=off -std=legacy"   # same no-FMA flag nwsrfsr uses
for f in types sorting stats utilities; do gfortran $F ../$S/$f.f90; done
for f in ../$S/*.f; do gfortran $F "$f"; done
gfortran $F ../$S/sac_snow.f90 ../$S/zero_brent.f90
gfortran -shared -o libsacsnow.so *.o
echo "built build/libsacsnow.so"
