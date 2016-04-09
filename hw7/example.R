prawns<-read.csv ('clean−prawns.csv', header=FALSE,skip = 1)
ndat<-dim(prawns)[1]
srange<-c(0.1,0.15,0.2,0.25,0.3,0.35)
xmat<-as.matrix(prawns[,c(1,2)])
spaces<-dist(xmat,method="euclidean",diag=FALSE,upper=FALSE)
msp<-as.matrix(spaces)
wmat<-exp(-msp/2*srange[1]^2)
for (i in 2:6) {
  grammat<-exp(-msp/(2*srange[i]^2))
  wmat<-cbind(wmat,grammat)
}
wmod<-cv.glmnet(wmat, as.vector(prawns[,3]), alpha=0.5)
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
wmat<-exp(-distsampletopts/(2*srange[1]^2))
for (i in 2:6) {
  grammmat<-exp(-distsampletopts/(2*srange[i]^2))
  wmat<-cbind(wmat, grammmat)
}
preds<-predict.cv.glmnet(wmod,wmat,s='lambda.min')
zmat<-matrix(0, nrow=100, ncol=100)
ptr<-1
for (i in 1:100) {
  for (j in 1:100) {
    zmat[i,j]<-preds[ptr]
    ptr<-ptr+1
  }
}
wscale=max(abs(min(preds)), abs(max(preds)))
image(yvec, xvec, (t(zmat)+wscale)/(2*wscale),xlab='Longtiude', ylab='Latitude', col=grey(seq(0,1,length=256)), useRaster=TRUE)
plot(wmod)
