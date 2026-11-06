.libPaths(c("~/R/lib", .libPaths()))
suppressMessages(library(nwsrfsr))
set.seed(7)
data(nrkw1_forcing); data(nrkw1_pars)
fa <- fa_nwrfc(6, nrkw1_forcing, nrkw1_pars)
nz <- length(fa)
cat("zones:", nz, " steps:", nrow(fa[[1]]), "\n")
dir.create("sac", showWarnings=FALSE)
for (z in 1:nz) write.csv(fa[[z]][,c("year","month","day","hour","map_mm","ptps","mat_degc","etd_mm")],
                          sprintf("sac/forcing_z%d.csv", z), row.names=FALSE)
# daily aggregation of the same forcing, to test the daily timestep used with Caravan
fd <- lapply(fa, function(x) {
  k <- paste(x$year, x$month, x$day)
  data.frame(year=tapply(x$year,k,`[`,1), month=tapply(x$month,k,`[`,1), day=tapply(x$day,k,`[`,1), hour=12,
             map_mm=tapply(x$map_mm,k,sum), ptps=tapply(x$ptps,k,mean), mat_degc=tapply(x$mat_degc,k,mean),
             etd_mm=tapply(x$etd_mm,k,sum))[order(as.Date(paste(tapply(x$year,k,`[`,1),tapply(x$month,k,`[`,1),tapply(x$day,k,`[`,1),sep="-"))),]
})
for (z in 1:nz) write.csv(fd[[z]], sprintf("sac/forcing_daily_z%d.csv", z), row.names=FALSE)
caps <- c("uztwm","uzfwm","lztwm","lzfpm","lzfsm")
allp <- list(); out <- list()
for (s in 0:10) {
  p <- as.data.frame(nrkw1_pars)
  if (s > 0) { i <- p$name %in% caps & p$type == "sac"; p$value[i] <- p$value[i] * runif(sum(i), 0.3, 3) }
  names_needed <- c("uztwm","uzfwm","lztwm","lzfpm","lzfsm","adimp","uzk","lzpk","lzsk","zperc","rexp","pctim","pfree","riva","side","rserv","efc",
                    "scf","mfmax","mfmin","uadj","si","nmf","tipm","mbase","plwhc","daygm","adc_a","adc_b","adc_c","alat","elev","peadj","pxadj","init_swe")
  w <- sapply(names_needed, function(n) p[p$name == n, ]$value)
  allp[[s+1]] <- cbind(set=s, zone=1:nz, as.data.frame(matrix(w, nrow=nz, dimnames=list(NULL, names_needed))))
  for (ts in c(6, 24)) {
    st <- sac_snow_states(ts, if (ts == 6) fa else fd, p)
    keep <- grep("^(uztwc|lztwc|lzfpc|swe|tci|aet)_", names(st), value=TRUE)
    out[[length(out)+1]] <- cbind(set=s, dt=ts, step=seq_len(nrow(st)), st[, keep])
  }
}
write.csv(do.call(rbind, allp), "sac/pars.csv", row.names=FALSE)
for (ts in c(6,24)) write.csv(do.call(rbind, Filter(function(o) o$dt[1]==ts, out)), sprintf("sac/nwsrfsr_out_dt%d.csv", ts), row.names=FALSE)
cat("nwsrfsr", as.character(packageVersion("nwsrfsr")), "done\n")
