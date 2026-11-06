# Reference unit-hydrograph routing from nwsrfsr::uh(), for checking sacsnow.uh() (run_uh_wrapper.py).
# Run where R + nwsrfsr are installed:  Rscript run_nwsrfsr_uh.R   -> writes ./uh/
.libPaths(c("~/R/lib", .libPaths()))
suppressMessages(library(nwsrfsr))
set.seed(11)
dir.create("uh", showWarnings = FALSE)

# Parameter sets (shape, scale in days); one "zone" per set. shape < 0 = no routing.
sets <- data.frame(shape = c(2, 1, 5, 1.5, 0.8, 3, 10, 1.2, -1),
                   scale = c(1, 0.5, 0.3, 3, 2, 0.2, 0.1, 6, 1))
nz <- nrow(sets)
n <- 3000
# intermittent synthetic channel inflow (mm per step), one column per zone
tci <- sapply(1:nz, function(z) ifelse(runif(n) < 0.3, rgamma(n, 0.7, scale = 8), 0))
write.csv(data.frame(step = 1:n, tci), "uh/tci.csv", row.names = FALSE)
write.csv(cbind(zone = 1:nz, sets), "uh/pars.csv", row.names = FALSE)

pars <- do.call(rbind, lapply(1:nz, function(z) data.frame(
  name = c("unit_shape", "unit_scale", "unit_toc", "unit_toc_adj", "zone_area"),
  zone = z, value = c(sets$shape[z], sets$scale[z], NA, NA, 1))))

out <- list()
for (dt in c(24, 6)) for (sot in c(TRUE, FALSE)) {
  cfs <- uh(dt, tci, pars, sum_zones = FALSE, start_of_timestep = sot, backfill = TRUE)
  mm <- cfs / (1000 * 3.28084^3 / (dt * 3600) * 1)  # undo nwsrfsr's cfs conversion (zone_area = 1)
  out[[length(out) + 1]] <- data.frame(dt = dt, start_of_timestep = sot, step = 1:n, mm)
}
write.csv(do.call(rbind, out), "uh/nwsrfsr_uh.csv", row.names = FALSE)
cat("nwsrfsr", as.character(packageVersion("nwsrfsr")), "done\n")
