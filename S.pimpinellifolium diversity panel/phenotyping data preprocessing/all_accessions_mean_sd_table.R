combinemean <- read.csv("C:/Users/Darui Yen/OneDrive/®à­±/combinemean.csv", sep = ",")
combinemean <- combinemean[,-1]
colnames(combinemean)[1] <- "accession"

PIref <- read.csv("C:/Users/Darui Yen/OneDrive/®à­±/crawler_pimpinellifolium_list.csv", sep = ",")
PIref$accession[161] <- "2017A02100"

combinemean$Fsd <- rep(NA, nrow(combinemean))
combinemean$Msd <- rep(NA, nrow(combinemean))
combinemean$Rsd <- rep(NA, nrow(combinemean))
combinemean$Esd <- rep(NA, nrow(combinemean))

Findex <- seq(3,49,2)
Mindex <- seq(2,48,2)


for (i in 1:nrow(combinemean)) {
  combinemean$Fsd[i] <- sd(combinemean[i,Findex], na.rm = T)
  combinemean$Msd[i] <- sd(combinemean[i,Mindex], na.rm = T)
  combinemean$Rsd[i] <- sd(combinemean[i,Mindex]/combinemean[i,Findex], na.rm = T)
  combinemean$Esd[i] <- sd(combinemean[i,Findex]-combinemean[i,Mindex], na.rm = T)
}

for (i in 50:57) {
  combinemean[,i] <- round(combinemean[,i], 3)
}

#####merge ID and accession#####
full <- merge(x = PIref[,c(1,4,7,8)], y = combinemean[,c(1,50:57)], by = "accession", all = TRUE)

full$F_mean_sd <- rep(NA, nrow(full))
full$M_mean_sd <- rep(NA, nrow(full))
full$R_mean_sd <- rep(NA, nrow(full))
full$E_mean_sd <- rep(NA, nrow(full))

###insert "+_" symbol###
for (i in 1:nrow(full)) {
  full$F_mean_sd[i] <- paste0(full$Fmean[i], "\u00b1", full$Fsd[i])
  full$M_mean_sd[i] <- paste0(full$Mmean[i], "\u00b1", full$Msd[i])
  full$R_mean_sd[i] <- paste0(full$MFratio[i], "\u00b1", full$Rsd[i])
  full$E_mean_sd[i] <- paste0(full$exsertion[i], "\u00b1", full$Esd[i])
}

write.csv(full, "C:/Users/Darui Yen/OneDrive/®à­±/all_mean_sd.csv")