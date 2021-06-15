setwd("C:/Users/Darui Yen/OneDrive/桌面/碩士論文/19TS-1外表型")
library(ggplot2)

data <- read.csv("19TS1st.csv", header = T, sep = ",")
colnames(data)[1] <- "accession"
data <- data[1:133,]
control <- subset(data, data[,1] == "606"|data[,1] == "619"|data[,1] == "631"|data[,1] == "639"|data[,1] == "659"|
                    data[,1] == "604"|data[,1] == "625"|data[,1] == "633"|data[,1] == "637"|data[,1] == "646")
rownames(control) <- control[,1]

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
shapiro.test(data[,19])#anther
shapiro.test(data[,18])#stigma
qqnorm(data[,20])
qqline(data[,20])
