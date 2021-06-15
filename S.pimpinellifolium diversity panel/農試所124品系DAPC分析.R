library(vcfR)
library(adegenet)
library(adegraphics)
library(pegas)
library(StAMPP)
library(lattice)
library(gplots)
library(ape)
library(ggmap) 


setwd("C:/Users/Darui Yen/Desktop")
vcf <- read.vcfR("QF08_SNPs_only.vcf.gz")

chrom <- create.chromR(name="2017A_tomato", vcf=vcf)

chrom1 <- masker(chrom, min_QUAL = 1, min_DP = 400, max_DP = 3000, min_MQ = 30,  max_MQ = 60)

write.vcf()

vcfnew <- read.vcfR("New_QF08.vcf.gz")
aa.genlight <- vcfR2genlight(vcfnew, n.cores=1) 

pca.1 <- glPca(aa.genlight, nf=500, n.cores=2)
pca.1$eig[1]/sum(pca.1$eig)


pop(aa.genlight)<-substr(indNames(aa.genlight),6,10) 

g1 <- s.class(pca.1$scores, pop(aa.genlight), xax=2, yax=1, col=transp(col,.6),
        ellipseSize=0, starSize=0, ppoints.cex=4, paxes.draw=T, pgrid.draw =F, plot = F) 
g2 <- s.label (pca.1$scores, xax=1, yax=2, ppoints.col = "red", plabels =
        list(box = list(draw = FALSE), optim = TRUE), paxes.draw=T, pgrid.draw =F, plabels.cex=1, plot = FALSE)
ADEgS(c(g1, g2), layout = c(1, 2)) 

grp <- find.clusters(aa.genlight, max.n.clust=10, glPca = pca.1, perc.pca = 100, n.iter=1e6, n.start=1000)
write.table(grp$grp, file="grouping_Kmeans_all.txt", sep="\t", quote=F, col.names=F) 

dapc1 <- dapc(aa.genlight, grp$grp)
scatter(dapc1, posi.da="bottomright", bg="white", scree.pca=TRUE, posi.pca="bottomleft")
