library(caret)
data<-read.csv("pima-indians-diabetes.data", header=FALSE)
x<-data[,-9]
y<-data[,9]
for (i in c(3,4,6,8)) {
  row<-x[,i]==0
  x[row,i]=NA
}
partition<-createDataPartition(y=y, p=0.8, list=FALSE)
trainX<-x[partition,]
trainY<-y[partition]
flag<-trainY==1
trainP<-trainX[flag,]
trainN<-trainX[!flag,]
pMeans<-sapply(trainP, mean, na.rm=T)
nMeans<-sapply(trainN, mean, na.rm=T)
pStd<-sapply(trainP, sd, na.rm=T)
nStd<-sapply(trainN, sd, na.rm=T)
# -(x-u)^2 / 2*sig^2 - log sig
testX<-x[-partition,]
testY<-y[-partition]
pOffset<-t(t(testX)-pMeans)
nOffset<-t(t(testX)-nMeans)
pScales<-t(t(pOffset) / pStd)
nScales<-t(t(nOffset) / nStd)
pLog<--(1/2)*rowSums(apply(pScales, c(1,2), function(x)x^2), na.rm = T) - sum(log(pStd))
nLog<--(1/2)*rowSums(apply(nScales, c(1,2), function(x)x^2), na.rm = T) - sum(log(nStd))
lvw=pLog>nLog
correct<-lvw==testY
accuracy<-sum(correct)/(sum(correct)+sum(!correct))
accuracy
