.libPaths(c("~/R/lib", .libPaths()))
suppressMessages(library(airGR))
set.seed(42)
data(L0123002)  # snow-influenced example catchment
P <- BasinObs$P; E <- BasinObs$E; T <- BasinObs$T; d <- BasinObs$DatesR
write.csv(data.frame(date=format(d,"%Y-%m-%d"),P=P,E=E,T=T), "forcing_L0123002.csv", row.names=FALSE)
write.csv(BasinInfo$HypsoData, "hypso_L0123002.csv", row.names=FALSE)
cat(BasinInfo$HypsoData[51], "\n", file="zinputs.txt")
IM <- CreateInputsModel(FUN_MOD=RunModel_CemaNeigeGR4J, DatesR=d, Precip=P, PotEvap=E, TempMean=T,
                        ZInputs=median(BasinInfo$HypsoData), HypsoData=BasinInfo$HypsoData, NLayers=5)
iw <- seq(which(format(d,"%Y-%m-%d")=="1990-01-01"), which(format(d,"%Y-%m-%d")=="1990-12-31"))
ir <- seq(which(format(d,"%Y-%m-%d")=="1991-01-01"), which(format(d,"%Y-%m-%d")=="2001-12-31"))
writeLines(as.character(c(min(iw),max(iw),min(ir),max(ir))-1), "periods.txt")
RO <- CreateRunOptions(FUN_MOD=RunModel_CemaNeigeGR4J, InputsModel=IM, IndPeriod_WarmUp=iw, IndPeriod_Run=ir)
n <- 25
pars <- cbind(X1=runif(n,50,1500), X2=runif(n,-3,2), X3=runif(n,20,400), X4=runif(n,0.6,4),
              C1=runif(n,0,1), C2=runif(n,1,8))
write.csv(pars, "params.csv", row.names=FALSE)
res <- list()
for (i in 1:n) {
  O <- RunModel_CemaNeigeGR4J(InputsModel=IM, RunOptions=RO, Param=pars[i,])
  swe <- Reduce(`+`, lapply(O$CemaNeigeLayers, function(l) l$SnowPack))/length(O$CemaNeigeLayers)
  res[[i]] <- data.frame(set=i, step=seq_along(O$Qsim), Qsim=O$Qsim, Prod=O$Prod, Rout=O$Rout, SWE=swe)
}
write.csv(do.call(rbind,res), "airgr_out.csv", row.names=FALSE)
cat("airGR", as.character(packageVersion("airGR")), "done\n")
