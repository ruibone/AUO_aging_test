group <- read.csv('C:/Users/Darui Yen/MEGAsync/agron_thesis/code/output/subgroup_anova.csv', sep = ',', header = T)
group$X <- NULL
group$subgroup[51] <- 'ecuador'
new_group = group[(group$subgroup != 'central america'),]

anova(lm(Mmean~subgroup, data = new_group))
anova(lm(Fmean~subgroup, data = new_group))
anova(lm(MFratio~subgroup, data = new_group))
anova(lm(exsertion~subgroup, data = new_group))
