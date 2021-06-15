library(ggplot2)
library("DataCombine")

#read data1 control
setwd("C:/Users/Darui Yen/OneDrive/桌面/碩士論文/19TS-2外表型")
data <- read.csv("19TS-2_phenotype.csv", header = T, sep = ",")
colnames(data1)[1] <- "accession"
data1 <- data1[1:133,]
control1 <- subset(data1, data1[,1] == "619"|data1[,1] == "606"|data1[,1] == "631"|data1[,1] == "639"|data1[,1] == "659"|
                    data1[,1] == "604"|data1[,1] == "625"|data1[,1] == "633"|data1[,1] == "637"|data1[,1] == "646")
rownames(control1) <- control1[,1]
rownames(control1) <- c("1946","1948","1964","1970","1976","1978","1982","1984","1991","2004")

#read data2 control
setwd("C:/Users/Darui Yen/OneDrive/桌面/碩士論文/19TS-2外表型")
data2 <- read.csv("19TS-2nd.csv", header = T, sep = ",")
control2 <- data2[1:9,]
control2 <- InsertRow(control2, rep(NA,20), 5)
control2[,21] <- c(1948,1964,1976,1984,2004,1946,1970,1978,1982,1991)
control2.order <- order(control2[,21])
control2 <- control2[control2.order,]
       

#exsertion comparison
control.ex <- data.frame(exsertion = c(control1[,20], control2[,20]), 
                         accession = factor(control2[,21]), batch = factor(rep(c("1st","2nd"), each = 10)))
ggplot(control.ex, aes(x = accession, y = exsertion, fill = batch)) +
  geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("                                                      Stigma  Exsertion") + 
  geom_text(x=10.25, y=0.006, label="NA")

#stamen comparison
control.stamen <- data.frame(stamen_length = c(control1[,18], control2[,18]), 
                         accession = factor(control2[,21]), batch = factor(rep(c("1st","2nd"), each = 10)))
ggplot(control.stamen, aes(x = accession, y = stamen_length, fill = batch)) +
  geom_bar(stat="identity", position=position_dodge()) +  ylab("stamen length") + ylim(c(0,0.65)) +
  ggtitle("                                                           Stamen Length") +
  geom_text(x=10.25, y=0.016, label="NA")

#stigma comparison
control.stigma <- data.frame(stigma_length = c(control1[,19], control2[,19]), 
                             accession = factor(control2[,21]), batch = factor(rep(c("1st","2nd"), each = 10)))
ggplot(control.stigma, aes(x = accession, y = stigma_length, fill = batch)) +
  geom_bar(stat="identity", position=position_dodge()) + ylab("stigma length") + ylim(c(0,0.65)) +
  ggtitle("                                                           Stigma Length") +
  geom_text(x=10.25, y=0.016, label="NA")

#----------------------------------------------------------------------------------------------------------------------
#phenotype order
order.control <- sort.int(control[,20],index.return = T)$ix
plot.control <- control[order.control,]
order.data <- sort.int(data[,20],index.return = T)$ix
plot.data <- data[order.data,]


#anther length
boxplot(t(plot.data[,c(2,4,6,8,10,12,14,16)]))
boxplot(t(plot.control[,c(2,4,6,8,10,12,14,16)]))
ggplot(plot.data , aes(plot.data[,19])) + geom_histogram(binwidth = 0.01) + xlab("stamen length")

#stigma length
boxplot(t(plot.data[,c(2,4,6,8,10,12,14,16)]))
boxplot(t(plot.control[,c(3,5,7,9,11,13,15,17)]))
ggplot(plot.data , aes(plot.data[,18])) + geom_histogram(binwidth = 0.01) + xlab("stigma length")

#exsertion
barplot(plot.data[,20], ylim = c(-0.1,0.2))
barplot(plot.control[,20], names.arg = plot.control[,1] )

#exsertion length count (normal distributed???????????????????????????????????????????????)
ggplot(plot.data , aes(plot.data[,20]), stat = ) + geom_histogram(binwidth = 0.01) + xlab("stigma exsertion")

#anther&stigma&exsertion
boxplot(data[,c(18,19,20)])

#normality test
shapiro.test(data[,20])
shapiro.test(data[,19])
shapiro.test(data[,18])
qqnorm(data[,20])
qqline(data[,20])
#---------------------------------------------------------------------------------------------------------
ggplot(plot.data18, aes(plot.data18[,18])) + geom_histogram(binwidth = 0.01)+ 
  xlab("stamen length") + ggtitle("binwidth = 0.01")

boxplot(data[,18:20])
barplot(plot.data20[,20])
                    