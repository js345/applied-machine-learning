efd<-as.matrix(read.table('./Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt',sep=",",header=FALSE))
x<-efd[,-(1:2)]
latitude<-efd[,ncol(efd)-1]
longitude<-efd[,ncol(efd)]
# linear regression - latitdue as y
y<-latitude
foo<-data.frame(x=x,y=y)
foo.lm<-lm(y~x,data=foo)
plot(y)
abline(foo.lm)