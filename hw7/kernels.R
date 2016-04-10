library(glmnet)
library(caret)
location<-read.table('./Locations.txt',sep=" ",header=TRUE)
temperature<-read.table('./Oregon_Met_Data.txt',sep=" ",header=TRUE)
# ignore data points of value 9999
temperature<-temperature[temperature$Tmin_deg_C!=9999,]
# avg(Tmin_deg_C) group by SID, Year -> annual mean temperature for each year
#result<-setNames(aggregate(temperature$Tmin_deg_C, list(temperature$SID,temperature$Year), mean),c("SID","Year","Tmin_deg_C"))
# avg(Tmin_deg_C) group by SID -> average annual mean temperature for each site
result<-setNames(aggregate(temperature$Tmin_deg_C, list(temperature$SID), mean),c("SID","Tmin_deg_C"))
# join location and result by SID
data<-merge(location,result, by="SID")
# into matrix
data<-as.matrix(data[,c('East_UTM','North_UTM','Tmin_deg_C')])
xmat<-data[,c(1,2)]
y<-as.vector(data[,3])
# training - cross validation for best scale
srange<-seq(90,140,10)
mses<-rep(0,length(srange))
flds<-createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
for (k in 1:5) {
  index<-1
  for (scale in srange) {
    # build the gram matrix with gaussian kernel
    xTrain<-xmat[flds[[k]],]
    xTest<-xmat[-flds[[k]],]
    yTrain<-y[flds[[k]]]
    yTest<-y[-flds[[k]]]
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
    mses[index]<-mses[index]+mse
    index<-index+1
  }
}
bestScale<-srange[which.min(mses)]
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
image(xvec, yvec, (zmat+wscale)/(2*wscale),xlab='East_UTM', ylab='North_UTM', col=heat.colors(20,alpha = 1), useRaster=TRUE, main="linear regression cross validated scales")

# part 2 regularized kernel methods
spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
msp<-as.matrix(spaces)
wmat<-exp(-msp/(2*srange[1]^2))
for (i in range(2:length(srange))) {
  grammat<-exp(-msp/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
lassomod<-cv.glmnet(wmat,y,alpha=1)
plot(lassomod,main="regularized lasso regression with gaussian kernels")
diff_ij<-function(i,j) sqrt(rowSums((pmat[i,]-xmat[j,])^2))
distsampletopts<-outer(seq_len(10000),seq_len(dim(xmat)[1]), diff_ij)
wmat<-exp(-distsampletopts/(2*srange[1]^2))
for (i in range(2:length(srange))) {
  grammat<-exp(-distsampletopts/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
preds<-predict.cv.glmnet(lassomod,wmat,lambda=lassomod$lambda.min)
zmat<-matrix(0, nrow=100, ncol=100)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image(xvec, yvec, (zmat+wscale)/(2*wscale),xlab='East_UTM', ylab='North_UTM', col=heat.colors(20,alpha = 1), useRaster=TRUE, main="lasso cross validated lambda")

# elastic net
spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
msp<-as.matrix(spaces)
wmat<-exp(-msp/(2*srange[1]^2))
for (i in range(2:length(srange))) {
  grammat<-exp(-msp/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
elastic<-cv.glmnet(wmat,y,alpha=0)
plot(elastic,main="regularized lasso regression with gaussian kernels")
diff_ij<-function(i,j) sqrt(rowSums((pmat[i,]-xmat[j,])^2))
distsampletopts<-outer(seq_len(10000),seq_len(dim(xmat)[1]), diff_ij)
wmat<-exp(-distsampletopts/(2*srange[1]^2))
for (i in range(2:length(srange))) {
  grammat<-exp(-distsampletopts/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
preds<-predict.cv.glmnet(lassomod,wmat,lambda=elastic$lambda.min)
zmat<-matrix(0, nrow=100, ncol=100)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image(xvec, yvec, (zmat+wscale)/(2*wscale),xlab='East_UTM', ylab='North_UTM', col=heat.colors(20,alpha = 1), useRaster=TRUE, main="alpha=0.5")