library(glmnet)
location<-read.table('./Locations.txt',sep=" ",header=TRUE)
temperature<-read.table('./Oregon_Met_Data.txt',sep=" ",header=TRUE)
# ignore data points of value 9999
temperature<-temperature[temperature$Tmin_deg_C!=9999,]
# avg(Tmin_deg_C) group by SID
result<-setNames(aggregate(temperature$Tmin_deg_C, list(temperature$SID), mean),c("SID","Tmin_deg_C"))
# join location and result by SID
data<-merge(location,result, by="SID")
# into matrix
data<-as.matrix(data[,c('East_UTM','North_UTM','Tmin_deg_C')])
xmat<-data[,c(1,2)]
y<-as.vector(data[,3])
n<-nrow(xmat)
split<-0.6
train<-sample(1:n, round(n*split))
xTrain<-xmat[train,]
xTest<-xmat[-train,]
yTrain<-y[train]
yTest<-y[-train]
# training - cross validation for best scale
srange<-c(40,45,50,55,60,65,70,80,90)
bestScale<-40
bestMSE<-2147483647
for (scale in srange) {
  # build the gram matrix with gaussian kernel
  spaces<-dist(xTrain,method="euclidean",diag=FALSE,upper=FALSE)
  msp<-as.matrix(spaces)
  wmat<-exp(-msp/(2*scale^2))
  wmod<-glmnet(wmat, yTrain, lambda=0)
  # test on held out data
  diff_ij<-function(i,j) sqrt(rowSums((xTest[i,]-xTrain[j,])^2))
  distsampletopts<-outer(seq_len(dim(xTest)[1]),seq_len(dim(xTrain)[1]), diff_ij)
  wmat<-exp(-distsampletopts/(2*scale^2))
  ypred<-predict.glmnet(wmod, wmat, lambda=0)
  # compute mse
  mse<-sum((yTest - ypred)^2) / nrow(xTest)
  print(scale)
  print(mse)
  if (mse < bestMSE) {
    bestScale<-scale
    bestMSE<-mse
  }
}
# use cv scale to train and predict
spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
msp<-as.matrix(spaces)
wmat<-exp(-msp/(2*bestScale^2))
wmod<-glmnet(wmat, y, lambda=0)
# 100 * 100 grid
xmin<-min(xmat[,1])
xmax<-max(xmat[,1])
ymin<-min(xmat[,2])
ymax<-max(xmat[,2])
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
wmat<-exp(-distsampletopts/(2*bestScale^2))
preds<-predict.glmnet(wmod, wmat, lambda=0)
zmat<-matrix(0, nrow=100, ncol=100)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image(xvec, yvec, (zmat+wscale)/(2*wscale),xlab='East_UTM', ylab='North_UTM', col=grey(seq(0, 1, length=512)), useRaster=TRUE)