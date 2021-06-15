library(dplyr)
library(ggplot2)

dat1 <- read.csv("C:/Users/碩士論文/19TS-1外表型/19TS1st_phenotype.csv", header = T, sep = ",")
dat2 <- read.csv("C:/Users/碩士論文/19TS-2外表型/19TS2nd_phenotype.csv", header = T, sep = ",")
dat3 <- read.csv("C:/Users/碩士論文/20TS-1外表型/20TS1st_phenotype_temperate.csv", header = T, sep = ",")
colnames(dat3)[1] <- "ID"
colnames(dat3)[21] <- "exsertion"

#filter----------------------------

dat2edit <- dat2[-c(1:9),]
dat1st <- rbind(dat1,dat2edit)
noseed <- dat1st[dat1st$ID == "2017A02039" | 
                   dat1st$ID == "2017A02040" | dat1st$ID == "2017A02053" | dat1st$ID == "2017A02080",]
dat1st <- dat1st[-as.numeric(rownames(noseed)),]
dat1avs2 <- merge(dat1st,dat3,by.x="ID",by.y="ID")
dat1in3 <- dat1[dat1$ID %in% dat3$ID,]

#control---------------------------

con1 <- dat1[dat1$ID == "2017A01946" | dat1$ID == "2017A01948" |  dat1$ID == "2017A01964" | dat1$ID == "2017A01970" | 
               dat1$ID == "2017A01976" | dat1$ID == "2017A01978" | dat1$ID == "2017A01982" | dat1$ID == "2017A01984" | 
               dat1$ID == "2017A01991" | dat1$ID == "2017A02004" ,]
con2 <- dat2[dat2$ID == "2017A01946" | dat2$ID == "2017A01948" |  dat2$ID == "2017A01964" | dat2$ID == "2017A01970" | 
               dat2$ID == "2017A01976" | dat2$ID == "2017A01978" | dat2$ID == "2017A01982" | dat2$ID == "2017A01984" | 
               dat2$ID == "2017A01991" | dat2$ID == "2017A02004" ,]
con3 <- dat3[dat3$ID == "2017A01946" | dat3$ID == "2017A01948" |  dat3$ID == "2017A01964" | dat3$ID == "2017A01970" | 
               dat3$ID == "2017A01976" | dat3$ID == "2017A01978" | dat3$ID == "2017A01982" | dat3$ID == "2017A01984" | 
               dat3$ID == "2017A01991" | dat3$ID == "2017A02004" ,]


#test between 1st and 3rd----------

t.test(dat1in3$exsertion,dat3$Mmean)
t.test(dat1in3$exsertion,dat3$Fmean)
t.test(dat1in3$exsertion,dat3$exsertion)

level <- c(rep("data1", nrow(dat1in3)), rep("data3", nrow(dat3)))
vecM <- c(dat1in3$Mmean, dat3$Mmean)
wilM <- data.frame(num = vecM, batch = level)
vecF <- c(dat1in3$Fmean, dat3$Fmean)
wilF <- data.frame(num = vecF, batch = level)
vecEx <- c(dat1in3$exsertion, dat3$exsertion)
wilEx <- data.frame(num = vecEx, batch = level)

wilcox.test(num~batch, data = wilM)
wilcox.test(num~batch, data = wilF)
wilcox.test(num~batch, data = wilEx)


#3rd batch ------------------------

shapiro.test(dat3$Mmean)
qqnorm(dat3$Mmean)
qqline(dat3$Mmean)
shapiro.test(dat3$Fmean)
qqnorm(dat3$Fmean)
qqline(dat3$Fmean)
shapiro.test(dat3$exsertion)
qqnorm(dat3$exsertion)
qqline(dat3$exsertion)

ggplot(dat3, aes(Mmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("stamen length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat3, aes(Fmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("style length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat3, aes(exsertion)) + geom_histogram(binwidth = 0.02)+ 
  xlab("stigma exsertion length") + ggtitle("                                                           binwidth = 0.02")


#other batch-----------------------

nrow(dat1[dat1$exsertion < 0,])
nrow(dat2[dat2$exsertion < 0,])
nrow(dat3[dat3$exsertion < 0,])

ggplot(dat1in3, aes(Mmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("stamen length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat1in3, aes(Fmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("style length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat1in3, aes(exsertion)) + geom_histogram(binwidth = 0.02)+ 
  xlab("exsertion length") + ggtitle("                                                             binwidth = 0.02")

ggplot(dat1st, aes(Mmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("stamen length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat1st, aes(Fmean)) + geom_histogram(binwidth = 0.02)+ 
  xlab("style length") + ggtitle("                                                             binwidth = 0.02")
ggplot(dat1st, aes(exsertion)) + geom_histogram(binwidth = 0.02)+ 
  xlab("exsertion length") + ggtitle("                                                             binwidth = 0.02")


#combine same accessions----------

aveM <- (dat1vs2$Mmean.x + dat1vs2$Mmean.y)/2
aveF <- (dat1vs2$Fmean.x + dat1vs2$Fmean.y)/2
aveEx <- (dat1vs2$exsertion.x + dat1vs2$exsertion.y)/2
combineframe <- data.frame(Mmean = aveM, Fmean = aveF, exsertion = aveEx)
boxplot(combineframe$Mmean ,combineframe$Fmean ,combineframe$exsertion,
        main = "Stamen & Style Comparison (64 accessions)", names = c("stamen","style","exsertion"))


#overall--------------------------

barplot(sort.int(dat1$exsertion), main = "1st batch", xlab = "exsertion length", ylim = c(-0.05,0.15))
barplot(sort.int(dat2$exsertion), main = "2nd batch", xlab = "exsertion length", ylim = c(-0.05,0.15))
barplot(sort.int(dat3$exsertion), main = "3rd batch", xlab = "exsertion length", ylim = c(-0.05,0.15))

boxplot(dat1$Mmean,dat2$Mmean,dat3$Mmean, main = "Stamen Comparison", names = c("1st","2nd","3rd"))
boxplot(dat1$Fmean,dat2$Fmean,dat3$Fmean, main = "Style Comparison", names = c("1st","2nd","3rd"))
boxplot(dat1$exsertion,dat2$exsertion,dat3$exsertion, main = "Exsertion Comparison", names = c("1st","2nd","3rd"))