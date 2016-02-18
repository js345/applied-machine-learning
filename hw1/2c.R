library(caret)
library(klaR)
data<-read.csv("pima-indians-diabetes.data", header=FALSE)
x<-data[,-9]
y<-data[,9]
partition<-createDataPartition(y=y, p=0.8, list=FALSE)
trainX<-x[partition,]
trainY<-y[partition]
trainY<-as.factor(trainY)
model<-train(trainX, trainY, method='nb', trControl = trainControl(method = 'cv', number=10))
teclasses<-predict(model,newdata=x[-partition,])
confusionMatrix(data=teclasses, reference=y[-partition])