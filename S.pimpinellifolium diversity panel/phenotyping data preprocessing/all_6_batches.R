library(dplyr)
library(magrittr)
library(ggplot2)
library(car)
library(reshape2)
library(plyr)
library(formattable)
library(bestNormalize)

#####read data#####
batch1 <- read.csv("C:/Users/agron_thesis/19TS-1_phenotype/19TS1st_phenotype.csv", header = T, sep = ",")
batch2 <- read.csv("C:/Users/agron_thesis/19TS-2_phenotype/19TS2nd_phenotype.csv", header = T, sep = ",")
batch3 <- read.csv("C:/Users/agron_thesis/20TS-1_phenotype/20TS1st_phenotype.csv", header = T, sep = ",")
batch4 <- read.csv("C:/Users/agron_thesis/20TS-2_phenotype/20TS2nd_phenotype.csv", header = T, sep = ",")
batch5 <- read.csv("C:/Users/agron_thesis/20TS-3_phenotype/20TS3rd_phenotype.csv", header = T, sep = ",")
batch6 <- read.csv("C:/Users/agron_thesis/21TS-1_phenotype/21TS1st_phenotype.csv", header = T, sep = ",")
colnames(batch3)[1] <- "ID"
colnames(batch4)[1] <- "ID"
colnames(batch3)[21] <- "exsertion"
colnames(batch4)[21] <- "exsertion"
colnames(batch5)[21] <- "exsertion"
colnames(batch6)[21] <- "exsertion"
batch6[batch6$ID == "2017A02100\t",]$ID <- "2017A02100"
batch6 <- batch6[1:117,]

for (i in 19:21) batch6[,i] <- as.numeric(batch6[,i])

########centimeter change to millimeter (old 0.1 centi = new 1 milli)&&&correct the scale (old 0.5 mm = new 1 mm)########

for (i in 3:21) {
  batch1[,i] <- batch1[,i]*20
  batch2[,i] <- batch2[,i]*20
  batch3[,i] <- batch3[,i]*20
  batch4[,i] <- batch4[,i]*20
  batch5[,i] <- batch5[,i]*20
  batch6[,i] <- batch6[,i]*20
}




#####data duplicate preprocessing#####
#首先將batch3中的缺值品系篩掉，並且確認這些品系在batch4中都有完整的資料
dupli1 <- which(batch3$ID %in% batch4$ID == 1)
batch3[dupli1,]$ID
#"2017A01948"為唯一存在於batch3的對照組(其他9個都在batch4)；"2017A01929" "2017A01933" "LA3638"為有缺值的品系
dupli1 <- dupli1[-3]
batch3 <- batch3[-dupli1,]#刪除有缺值且重複的品系後，dim = 61 21

#將兩兩batch合併為一replicate
repli1 <- rbind(batch1, batch2)
repli2 <- rbind(batch3, batch4)
repli3 <- rbind(batch5, batch6)

#接著找replicate2中重複或缺失的品系
dupli2 <- which(duplicated(repli2$ID) == 1)
repli2[dupli2,]$ID
#"2017A01946" "2017A01948" "2017A01964" "2017A01970" "2017A01976" "2017A01978" "2017A01982" "2017A01984"
#"2017A01991" "2017A02004"為對照組
batch4[batch4$ID == "LA1617",]
#"LA1617"為batch4中重複種植的品系
batch4 <- batch4[-28,]#刪除重複種植品系中的第一筆資料後，dim = 183 21
repli2 <- rbind(batch3, batch4) # dim = 244 21

#再來找replicate2中重複或缺失的品系
dupli3 <- which(repli1$ID %in% repli2$ID  == 0)
repli1[dupli3,]$ID
#"2017A02039" "2017A02053" "2017A02080"沒有收到種子；"LA1587"在replicate2沒有種到(batch4中沒有種到)
batch2 <- batch2[-c(45,57,83),]#刪除三個沒種子的品系，dim = 111 21
#batch1 dim = 133 21 
repli1 <- rbind(batch1, batch2) #dim = 244 21

#確認replicate1中重複或缺失的品系
dupli4 <- which(duplicated(repli1$ID) == 1)
repli1[dupli4,]$ID
#"2017A01948" "2017A01964" "2017A01976" "2017A01984" "2017A02004" "2017A01946" "2017A01970" "2017A01978" "2017A01982"
#以上9個皆為對照組，表示在replicate1中(batch2中長不好收不到花)還少了一個對照組"2017A01991"

#新增batch5，dim = 125 21 , batch6 dim = 117 21
#確認replicate3中缺失的品系
dupli5 <- which(repli2$ID %in% repli3$ID == 0)
repli2[dupli5,]$ID
#2017A2042" 2017A2087" "2017A2088"，種子短缺

#missing rate
missing <- (sum(is.na(rbind(repli1[,3:18], repli2[,3:18], repli3[,3:18])))) / (16*(245+244+242))

#####control group#####

index1<- index2 <- index3 <- index4 <- index5 <- index6 <- 0
controlname <- c("2017A01946", "2017A01948", "2017A01964", "2017A01970", "2017A01976", 
                 "2017A01978", "2017A01982", "2017A01984","2017A01991", "2017A02004")
for (i in 1:10) {
  index1 <- c(which(batch1$ID == controlname[i]), index1)
  index2 <- c(which(batch2$ID == controlname[i]), index2)
  index3 <- c(which(batch3$ID == controlname[i]), index3)
  index4 <- c(which(batch4$ID == controlname[i]), index4)
  index5 <- c(which(batch5$ID == controlname[i]), index5)
  index6 <- c(which(batch6$ID == controlname[i]), index6)
} 

control1 <- batch1[index1,c(1,19:21)]
control2 <- batch2[index2,c(1,19:21)]
control3 <- rbind(batch3[index3,c(1,19:21)], batch4[index4[index4 < 50],c(1,19:21)])
control4 <- batch4[index4[index4 > 50],c(1,19:21)]
control5 <- batch5[index5,c(1,19:21)]
control6 <- batch6[index6,c(1,19:21)]
control <- rbind(control1, control2, control3, control4, control5, control6) %>% 
  mutate(batch = c(rep("1st", nrow(control1)), rep("2nd", nrow(control2)), rep("3rd", nrow(control3)), 
                   rep("4th", nrow(control4)), rep("5th", nrow(control5)), rep("6th", nrow(control6))))

widecontrol <- full_join(control1, control2, by = "ID") %>%
  full_join(control3, by = "ID") %>%
  full_join(control4, by = "ID") %>%
  full_join(control5, by = "ID") %>%
  full_join(control6, by = "ID")
controlM <- widecontrol[,seq(2, ncol(widecontrol),3)]
controlF <- widecontrol[,seq(3, ncol(widecontrol),3)]
controle <- widecontrol[,seq(4, ncol(widecontrol),3)]
control_mean_sd <- data.frame(ID = widecontrol$ID, 
                              Mmean = apply(controlM, 1, mean, na.rm = T), Fmean = apply(controlF, 1, mean, na.rm = T),
                              emean = apply(controle, 1, mean, na.rm = T), Msd = apply(controlM, 1, sd, na.rm = T),
                              Fsd = apply(controlF, 1, sd, na.rm = T), esd = apply(controle, 1, sd, na.rm = T))
control_mean_sd <- control_mean_sd[order(control_mean_sd$emean),]

longcontrol <- melt(control_mean_sd, id.vars = c("ID", "Msd", "Fsd"), measure.vars = c("Mmean", "Fmean"), 
                    variable.name = "gender", value.name = "length") %>%
  mutate(sd = c(control_mean_sd$Msd, control_mean_sd$Fsd))

barplot(sort(control1$exsertion), main = "Stigma Exsertion in Batch 1", ylim = c(-1, 3), xlab = "accession",
        names.arg  = substr(control1$ID[order(control1$exsertion)], start = 7, stop = 10), ylab = "length (mm)")
barplot(sort(control2$exsertion), main = "control2_exsertion", ylim = c(-1, 3),
        names.arg  = substr(control2$ID[order(control2$exsertion)], start = 7, stop = 10))
barplot(sort(control3$exsertion), main = "control3_exsertion", ylim = c(-1, 3),
        names.arg  = substr(control3$ID[order(control3$exsertion)], start = 7, stop = 10))
barplot(sort(control4$exsertion), main = "control4_exsertion", ylim = c(-0.05, 0.15),
        names.arg  = substr(control4$ID[order(control4$exsertion)], start = 7, stop = 10))
barplot(sort(control5$exsertion), main = "control5_exsertion", ylim = c(-0.05, 0.15),
        names.arg  = substr(control5$ID[order(control5$exsertion)], start = 7, stop = 10))
barplot(sort(control6$exsertion), main = "control6_exsertion", ylim = c(-0.05, 0.15),
        names.arg  = substr(control6$ID[order(control6$exsertion)], start = 7, stop = 10))

ggplot(data = control_mean_sd, aes(x = ID, y = Mmean)) + 
  geom_bar(stat = "identity", fill = "royalblue2", alpha = 0.6) +
  geom_errorbar(aes(x = ID, ymin = Mmean-Msd, ymax = Mmean+Msd), width = 0.3, colour="orange", alpha = 0.9, size = 1.5) +
  ggtitle("stamen length of control group") + scale_x_discrete(limits = control_mean_sd$ID)
ggplot(data = control_mean_sd, aes(x = ID, y = Fmean)) + 
  geom_bar(stat = "identity", fill = "royalblue2", alpha = 0.6) +
  geom_errorbar(aes(x = ID, ymin = Fmean-Fsd, ymax = Fmean+Fsd), width = 0.3, colour="orange", alpha = 0.9, size = 1.5) +
  ggtitle("style length of control group") + scale_x_discrete(limits = control_mean_sd$ID)
ggplot(data = control_mean_sd, aes(x = ID, y = emean)) + ylab("length (mm)") + xlab("accession") +
  geom_bar(stat = "identity", fill = "royalblue2", alpha = 0.6) +
  geom_errorbar(aes(x = ID, ymin = emean-esd, ymax = emean+esd), width = 0.3, colour="orange", alpha = 0.9, size = 1.5) +
  ggtitle("Stigma Exsertion of Control Groups") + scale_x_discrete(limits = control_mean_sd$ID)

ggplot(data = longcontrol, aes(x = ID, y = length, fill = gender)) + ylab("length (mm)") +
  geom_bar(stat = "identity", position = position_dodge()) + ggtitle("Stamen & Style Length of Control Groups") +
  geom_errorbar(aes(x = ID, ymin = length-sd, ymax = length+sd), position = position_dodge(), 
                width = 0.9, color="brown", alpha = 1, size = 1) + scale_x_discrete(limits = control_mean_sd$ID)

ggplot(data = control, aes(x = ID, y = Mmean, fill = batch)) + 
  geom_bar(stat = "identity", position = position_dodge()) + ggtitle("control_stamen")
ggplot(data = control, aes(x = ID, y = Fmean, fill = batch)) + 
  geom_bar(stat = "identity", position = position_dodge()) + ggtitle("control_style")
ggplot(data = control, aes(x = ID, y = exsertion, fill = batch)) + 
  geom_bar(stat = "identity", position = position_dodge()) + ggtitle("control_exsertion")

#####control test#####

anova(lm(Mmean~batch, data = control))
anova(lm(Fmean~batch, data = control))
anova(lm(exsertion~batch, data = control))

#####replicate group#####

index7 <- index8 <- index9 <- c()
controlname <- c("2017A01946", "2017A01948", "2017A01964", "2017A01970", "2017A01976", 
                 "2017A01978", "2017A01982", "2017A01984","2017A01991", "2017A02004")
for (i in 1:10) {
  index7 <- c(which(repli1$ID == controlname[i]), index7)
  index8 <- c(which(repli2$ID == controlname[i]), index8)
  index9 <- c(which(repli3$ID == controlname[i]), index9)
} 
index7 <- index7[index7 > nrow(batch1)]#將batch2中的對照組刪除
index8 <- index8[index8 <= 100]#將batch3中的對照組以及batch4中前面一組(補種的)對照組刪除
index9 <- index9[index9 >= 200]

comfirst <- repli1[-index7,c(1,3:18)]#dim = 235 17
comsecond <- repli2[-index8,c(1,3:18)]#dim = 234 17
comthird <- repli3[-index9,c(1,3:18)]#dim = 232 17

first <- repli1[-index7,c(1,19:21)]#dim = 235 4
second <- repli2[-index8,c(1,19:21)]#dim = 234 4
third <- repli3[-index9,c(1,19:21)]#dim = 232 4


#####replicate distribution#####

#barplot
barplot(sort(first$exsertion), main = "exsertion_1st", ylim  = c(-0.1, 0.15), 
        xlab = paste("exsertion rate = ", round(sum(first$exsertion >= 0)/length(first$exsertion),4)))
#exsertion rate = 217/235 = 0.9234
barplot(sort(second$exsertion), main = "exsertion_2nd", ylim  = c(-0.05, 0.2),
        xlab = paste("exsertion rate = ", round(sum(second$exsertion >= 0)/length(second$exsertion),4)))
#exsertion rate = 215/234 = 0.9188
barplot(sort(third$exsertion), main = "exsertion_3rd", ylim  = c(-0.05, 0.2),
        xlab = paste("exsertion rate = ", round(sum(third$exsertion >= 0, na.rm = T)/length(third$exsertion),4)))
#exsertion rate = 214/232 = 0.9224

all_exsertion <- (217+215+214)/(235+234+232)

#ggplot preprocessing
longfirst <- melt(first, id.vars = c("ID", "exsertion"), measure.vars = c("Mmean", "Fmean"), 
                  variable.name = "gender", value.name = "length")
longsecond <- melt(second, id.vars = c("ID", "exsertion"), measure.vars = c("Mmean", "Fmean"), 
                   variable.name = "gender", value.name = "length")
longthird <- melt(third, id.vars = c("ID", "exsertion"), measure.vars = c("Mmean", "Fmean"), 
                  variable.name = "gender", value.name = "length")
longcombined <- rbind(first, second, third) %>% 
  mutate(replicate = c(rep("1st", nrow(first)), rep("2nd", nrow(second)), rep("3rd", nrow(third))))

mu1 <- ddply(longfirst, "gender", summarise, grp.mean = mean(length))
mu2 <- ddply(longsecond, "gender", summarise, grp.mean = mean(length))
mu3 <- ddply(longthird, "gender", summarise, grp.mean = mean(length, na.rm = T))
mu4 <- ddply(longcombined, "replicate", summarise, grp.mean = mean(Mmean, na.rm = T))
mu5 <- ddply(longcombined, "replicate", summarise, grp.mean = mean(Fmean, na.rm = T))
mu6 <- ddply(longcombined, "replicate", summarise, grp.mean = mean(exsertion, na.rm = T))

#ggplot
ggplot(longfirst, aes(x = length, fill = gender)) + geom_density(alpha = 0.4) + xlim(2,7.5) +
  geom_vline(data = mu1, aes(xintercept = grp.mean, color = gender), linetype = "dashed", size = 2) +
  ggtitle("the stamen & style length of 1st replicate")
ggplot(longsecond, aes(x = length, fill = gender)) + geom_density(alpha = 0.4) + xlim(2,8) +
  geom_vline(data = mu2, aes(xintercept = grp.mean, color = gender), linetype = "dashed", size = 2) +
  ggtitle("the stamen & style length of 2nd replicate")
ggplot(longthird, aes(x = length, fill = gender)) + geom_density(alpha = 0.4) + xlim(2,8) +
  geom_vline(data = mu3, aes(xintercept = grp.mean, color = gender), linetype = "dashed", size = 2) +
  ggtitle("the stamen & style length of 3rd replicate")

ggplot(longcombined, aes(x = Mmean, fill = replicate)) + geom_density(alpha=0.4) + xlim(4,14) + 
  geom_vline(data = mu4, aes(xintercept = grp.mean, color = replicate), linetype = "dashed", size = 2) +
  theme(legend.position = "top") + xlab("length (mm)") + ggtitle("Stamen Length")
ggplot(longcombined, aes(x = Fmean, fill = replicate)) + geom_density(alpha=0.4) + xlim(4,16)  +
  geom_vline(data = mu5, aes(xintercept = grp.mean, color = replicate), linetype = "dashed", size = 2) +
  theme(legend.position = "top") + xlab("length (mm)") + ggtitle("Style Length")
ggplot(longcombined, aes(x = exsertion, fill = replicate)) + geom_density(alpha=0.4) + xlim(-2,4) + 
  ggtitle("Stigma Exsertion Length") + xlab("length (mm)") + 
  geom_vline(data = mu6, aes(xintercept = grp.mean, color = replicate), linetype = "dashed", size = 2) +
  theme(legend.position = "top")


###combine all 3 replicate to generate the mean of 235 accessions###

combinemean <- full_join(comfirst, comsecond, by = "ID") %>%
  full_join(comthird, by = "ID") 

Findex <- seq(3, ncol(combinemean), 2)
Mindex <- seq(2, ncol(combinemean), 2)

combinemean <- combinemean %>%
  mutate(Fmean = rowMeans(combinemean[,Findex], na.rm = T), 
         Mmean = rowMeans(combinemean[,Mindex], na.rm = T)) 
combinemean <- mutate(combinemean, exsertion = combinemean$Fmean - combinemean$Mmean) #dim = 235 52(1+3*16+3)

#transfer the format for plotting
longFM <- melt(combinemean, id.vars = "ID", measure.vars = c("Mmean", "Fmean"), 
               variable.name = "gender", value.name = "length") 
for (i in 1:nrow(longFM)) {
  if (longFM$gender[i] == "Mmean") longFM$Gender[i] = "male" else
    longFM$Gender[i] = "female"
}


FMmean <- tapply(longFM$length, longFM$gender, mean)
mu7 <- data.frame("Gender" = c("male","female"), "grp.mean" = FMmean)
mu8 <- data.frame("grp.mean" = mean(combinemean$exsertion, na.rm = T))

#exsertion ratio
mu9 <- data.frame("grp.mean" = mean(combinemean$exsertion/combinemean$Fmean, na.rm = T))

ggplot(longFM, aes(x = length, fill = Gender)) + geom_density(alpha=0.4) + xlim(4,15) +
  geom_vline(data = mu7, aes(xintercept = grp.mean, color = Gender), linetype = "dashed", size = 2) +
  ggtitle("Stamen & Style Length of 235 accessions") + xlab("length (mm)")
ggplot(combinemean, aes(x = exsertion)) + geom_density(alpha=0.4, fill = "lightsteelblue2") + xlim(-2.2,4.2) + 
  ggtitle("Stigma Exsertion of 235 accessions") +
  geom_vline(data = mu8, aes(xintercept = grp.mean), color = "royalblue4", linetype = "dashed", size = 2) + 
  theme(legend.position = "top") + xlab("length (mm)")

#exsertion ratio
ratio <- data.frame(x = (combinemean$exsertion/combinemean$Fmean))
ggplot(ratio, aes(x = x)) + geom_density(alpha=0.4, fill = "steelblue") + xlim(-0.2,0.35) + 
  ggtitle("the exsertion ratio of 235 accessions") + xlab("exsertion ratio") +
  geom_vline(data = mu9, aes(xintercept = grp.mean), color = "royalblue4", linetype = "dashed", size = 2) + 
  theme(legend.position = "top")

###output the allmean csv###

write.csv(combinemean, "C:/Users/Darui Yen/OneDrive/桌面/combinemean.csv")

#shapiro test
sha11 <- shapiro.test(first$Mmean)#p-value = 0.0001535
sha12 <- shapiro.test(first$Fmean)#p-value = 9.896e-08
sha13 <- shapiro.test(first$exsertion)#p-value = 8.017e-06
sha21 <- shapiro.test(second$Mmean)#p-value = 6.391e-08
sha22 <- shapiro.test(second$Fmean)#p-value = 1.049e-09
sha23 <- shapiro.test(second$exsertion)#p-value = 3.954e-05
sha31 <- shapiro.test(third$Mmean)#p-value = 6.943e-06
sha32 <- shapiro.test(third$Fmean)#p-value = 2.737e-08
sha33 <- shapiro.test(third$exsertion)#p-value = 0.0001316
sha01 <- shapiro.test(combinemean$Mmean)
sha02 <- shapiro.test(combinemean$Fmean)
sha03 <- shapiro.test(combinemean$exsertion)

#qqplot
par(mfrow = c(4,3))
qqPlot(first$Mmean, envelope = F)
title("stamen_1st")
qqPlot(first$Fmean, envelope = F)
title("style_1st")
qqPlot(first$exsertion, envelope = F)
title("exsertion_1st")
qqPlot(second$Mmean, envelope = F)
title("stamen_2nd")
qqPlot(second$Fmean, envelope = F)
title("style_2nd")
qqPlot(second$exsertion, envelope = F)
title("exsertion_2nd")
qqPlot(third$Mmean, envelope = F)
title("stamen_3rd")
qqPlot(third$Fmean, envelope = F)
title("style_3rd")
qqPlot(third$exsertion, envelope = F)
title("exsertion_3rd")
qqPlot(combinemean$Mmean, envelope = F)
title("stamen_combined")
qqPlot(combinemean$Fmean, envelope = F)
title("style_combined")
qqPlot(combinemean$exsertion, envelope = F)
title("exsertion_combined")
title("QQ plot", outer = T)

par(mfrow = c(1,1))

#####normalized phenotype#####

bestM <- bestNormalize(combinemean$Mmean, allow_orderNorm = T, out_of_sample = F)
bestF <- bestNormalize(combinemean$Fmean, allow_orderNorm = T, out_of_sample = F)
beste <- bestNormalize(combinemean$exsertion, allow_orderNorm = T, out_of_sample = F)

boxM <- boxcox(combinemean$Mmean)
boxF <- boxcox(combinemean$Fmean)

yeoM <- yeojohnson(combinemean$Mmean)
yeoF <- yeojohnson(combinemean$Fmean)
yeoE <- yeojohnson(combinemean$exsertion)

sha41 <- shapiro.test(bestM$x.t)
sha42 <- shapiro.test(bestF$x.t)
sha43 <- shapiro.test(beste$x.t)

shabM <- shapiro.test(boxM$x.t)
shabF <- shapiro.test(boxF$x.t)
shayM <- shapiro.test(yeoM$x.t)
shayF <- shapiro.test(yeoF$x.t)
shayE <- shapiro.test(yeoE$x.t)

shapiro_table <- data.frame(stamen = c(sha11$p.value, sha21$p.value, sha31$p.value, sha01$p.value, 
                                       shabM$p.value, shayM$p.value, sha41$p.value),
                            style = c(sha12$p.value, sha22$p.value, sha32$p.value, sha02$p.value, 
                                      shabF$p.value, shayF$p.value, sha42$p.value),
                            exsertion = c(sha13$p.value, sha23$p.value, sha33$p.value, sha03$p.value, 
                                          NA, shayE$p.value, sha43$p.value))
rownames(shapiro_table) <- c("replicate1", "replicate2", "replicate3", "overall", "boxcox", "yeojohnson", "transform")
formattable(shapiro_table)


layout_mat <- matrix(c(1:5,0,6:11), ncol = 3, nrow = 4, byrow = T)
par(mfrow = c(4,3))
layout(mat = layout_mat)

qqPlot(combinemean$Mmean, envelope = F, ylab = "data quantiles", id = F)
title("stamen length")
qqPlot(combinemean$Fmean, envelope = F, ylab = "data quantiles", id = F)
title("style length")
qqPlot(combinemean$exsertion, envelope = F, ylab = "data quantiles", id = F)
title("exsertion length")
qqPlot(boxM$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = "method = boxcox")
qqPlot(boxF$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = "method = boxcox")
qqPlot(yeoM$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = "method = yeojohnson")
qqPlot(yeoF$x.t, envelope = F, ylab = "data quantiles", id = F)
title("style length", sub = "method = yeojohnson")
qqPlot(yeoE$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = "method = yeojohnson")
qqPlot(bestM$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = paste("method = ", names(bestM$norm_stats)[which.min(bestM$norm_stats)]))
qqPlot(bestF$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = paste("method = ", names(bestF$norm_stats)[which.min(bestF$norm_stats)]))
qqPlot(beste$x.t, envelope = F, ylab = "data quantiles", id = F)
title(sub = paste("method = ", names(beste$norm_stats)[which.min(beste$norm_stats)]))
title("QQ plot", outer = T)


layout(matrix(c(1,2,3,3),2,2,byrow = T), width = c(1,1), height = c(1,2))

plot(density(bestM$x.t), main = "Stamen Length", lwd = 3, col = "red")
plot(density(bestF$x.t), main = "Style Length", lwd = 3, col = "red")
plot(density(beste$x.t), main = "Exsertion Length", lwd = 3, col = "red")
title("Quantile Transformation", outer = T)

par(mfrow = c(1,1))


transform_combinemean <- data.frame(trans_M = bestM$x.t, trans_F = bestF$x.t, trans_e = beste$x.t)
write.csv(transform_combinemean, "C:/Users/Darui Yen/OneDrive/桌面/transform_combinemean.csv")
