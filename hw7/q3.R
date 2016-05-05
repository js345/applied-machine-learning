library(glmnet)
library(caret)
location<-read.table('./Locations.txt',sep=" ",header=TRUE)
temperature<-read.table('./Oregon_Met_Data.txt',sep=" ",header=TRUE)
# ignore data points of value 9999
temperature<-temperature[temperature$Tmin_deg_C!=9999,]
# avg(Tmin_deg_C) group by SID, Year -> annual mean temperature for each year
# avg(Tmin_deg_C) group by SID -> average annual mean temperature for each site
result<-setNames(aggregate(temperature$Tmin_deg_C, list(temperature$SID), mean),c("SID","Tmin_deg_C"))
# join location and result by SID
data<-merge(location,result, by="SID")
# into matrix
data<-as.matrix(data[,c('East_UTM','North_UTM','Tmin_deg_C')])
xmat<-data[,c(1,2)]/1000
y<-as.vector(data[,3])
# training - cross validation for best scale
srange<-c(50,52,54,57,58,60)
# use cv scale to train and predict
spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
msp<-as.matrix(spaces^2)
wmat<-exp(-msp/(2*srange[1]^2))
for (i in 2:length(srange)) {
  grammat<-exp(-msp/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
wmod<-cv.glmnet(wmat, y, alpha=0.2)
plot(wmod,main="elastic")
# 100 * 100 grid
xmin<-min(xmat[,1])
xmax<-max(xmat[,1])
ymin<-min(xmat[,2])
ymax<-max(xmat[,2])
#xmin<-560000
#xmax<-660000
#ymin<-4900000
#ymax<-5000000
xvec<-seq(xmin,xmax,length=100)
yvec<-seq(ymin,ymax,length=100)
# these are the points
pmat<-matrix(0,nrow=100*100, ncol=2)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    pmat[ptr,1]<-xvec[i]
    pmat[ptr,2]<-yvec[j]
    ptr<-ptr+1
  }
}
diff_ij<-function(i,j) sqrt(rowSums((pmat[i,]-xmat[j,])^2))
distsampletopts<-outer(seq_len(10000),seq_len(dim(xmat)[1]), diff_ij)
wmat<-exp(-distsampletopts^2/(2*srange[1]^2))
for (i in 2:length(srange)) {
  grammat<-exp(-distsampletopts^2/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
preds<-predict.cv.glmnet(wmod, wmat, s='lambda.min')
zmat<-matrix(0, nrow=100, ncol=100)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image.plot(xvec, yvec, zmat ,xlab='East_UTM', ylab='North_UTM', col=heat.colors(20,alpha = 1), useRaster=TRUE, legend.only=FALSE, main="elastic alpha = 0.2")
