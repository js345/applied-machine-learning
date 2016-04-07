library(glmnet)
location<-read.table('./Locations.txt',sep=" ",header=TRUE)
temperature<-read.table('./Oregon_Met_Data.txt',sep=" ",header=TRUE)