library(glmnet)
library(caret)
location<-read.table('./Locations.txt',sep=" ",header=TRUE)
temperature<-read.table('./Oregon_Met_Data.txt',sep=" ",header=TRUE)
# ignore data points of value 9999
temperature<-temperature[temperature$Tmin_deg_C!=9999,]
# join tables by SID
result<-merge(location[,c("SID","East_UTM","North_UTM")], temperature[,c("SID","Time","Tmin_deg_C")], by="SID")

# 100 * 100 grid
xmin<-min(result$East_UTM)
xmax<-max(result$East_UTM)
ymin<-min(result$North_UTM)
ymax<-max(result$North_UTM)
xvec<-seq(xmin,xmax,length=100)
yvec<-seq(ymin,ymax,length=100)
preds<-rep(0,10000)
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
zmat<-matrix(0, nrow=100, ncol=100)

# build tables by time (days)
alldata<-list()
for (i in 1:1827) {
  tmp<-result[result$Time==i,]
  alldata[[i]]<-tmp[,c("East_UTM","North_UTM","Tmin_deg_C")]
  
  # train on this day
  data<-alldata[[i]]
  data<-as.matrix(data[,c('East_UTM','North_UTM','Tmin_deg_C')])
  xmat<-data[,c(1,2)]
  y<-as.vector(data[,3])
  # all ranges
  spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
  msp<-as.matrix(spaces)
  wmat<-exp(-msp/(2*srange[1]^2))
  for (i in 2:length(srange)) {
    grammat<-exp(-msp/(2*srange[i]^2))
    wmat<-cbind(wmat,grammat)
  }
  elastic<-cv.glmnet(wmat,y,alpha=0.5)
  
  diff_ij<-function(i,j) sqrt(rowSums((pmat[i,]-xmat[j,])^2))
  distsampletopts<-outer(seq_len(10000),seq_len(dim(xmat)[1]), diff_ij)
  wmat<-exp(-distsampletopts/(2*srange[1]^2))
  for (i in 2:length(srange)) {
    grammat<-exp(-distsampletopts/(2*srange[i]^2))
    wmat<-cbind(wmat,grammat)
  }
  pred<-predict.cv.glmnet(elastic,wmat,lambda=elastic$lambda.min)
  preds<-preds+pred
}
preds<-preds/1827
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image(xvec, yvec, (zmat+wscale)/(2*wscale),xlab='East_UTM', ylab='North_UTM', col=heat.colors(20,alpha = 1), useRaster=TRUE, main="lasso cross validated lambdas")