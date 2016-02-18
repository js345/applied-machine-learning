library(caret)
library(klaR)
data<-read.csv("pima-indians-diabetes.data", header=FALSE)
x<-data[,-9]
y<-data[,9]
partition<-createDataPartition(y=y, p=0.8, list=FALSE)
trainX<-x[partition,]
trainY<-y[partition]
model<-svmlight(x=trainX, grouping=trainY, pathsvm="/Users/xiaofo/svm_light")
teclasses<-predict(model,newdata=x[-partition,])
labels<-teclasses['class']
confusionMatrix(data=labels[[1]], reference=y[-partition])
