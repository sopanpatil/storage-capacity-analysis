# Third-party model code

Unchanged copies of two codes that have no archived release of their own, kept here so that
the exact versions used stay available. Each stays under the licence of its authors, not the
MIT licence of this repository.

| | `grsuite-1.1.1/` | `nwsrfsr-1.0.3/` |
|---|---|---|
| What it provides | GR6J-CemaNeige and GR4J-CemaNeige | SAC-SMA, SNOW-17 and the gamma unit hydrograph (NWS Fortran) |
| Upstream | [Najim33/GRsuite](https://github.com/Najim33/GRsuite), tag `v1.1.1`, commit `9568d13b98d3f0e86a2e8b00f30cd2f4c039ccd8` | CRAN package `nwsrfsr` 1.0.3, from the mirror [cran/nwsrfsr](https://github.com/cran/nwsrfsr), commit `8b6c85c43cc2b2d92649dd3374963d36de032a28` |
| Licence | GPL-2.0-or-later (`LICENSE`, `NOTICE`) | Apache License (>= 2) for the R package; US Government work; some MIT-licensed routines listed in `inst/COPYRIGHTS` |
| How it is used | Installed from PyPI (`requirements.txt`). This copy is not imported; it is the package source, for the record | `models/sacsma/build.sh` compiles `src/` into the library that `models/sacsma/sacsnow.py` calls |
| What is copied | `src/grsuite/` and the top-level licence, notice, README, changelog, citation and `pyproject.toml`. Upstream's tests, benchmarks, docs, examples and tools are left out | The whole package |

The source of `nwsrfsr` 1.0.3 is also identical, apart from line endings and final newlines,
to the code at commit `f7dd0905d64f9109fb62589ceca10241e310c867` of
[NOAA-NWRFC/nwsrfs-hydro-models](https://github.com/NOAA-NWRFC/nwsrfs-hydro-models).
`nwsrfsr-notices/` holds that repository's `LICENSE` and `NOTICE`, and the package's
`LICENSE.note`, which the CRAN package does not carry.

The installed `grsuite` 1.1.1 from PyPI is identical, file for file, to `src/grsuite/` at the
tag above.
