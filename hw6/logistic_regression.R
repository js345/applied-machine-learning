library(glmnet)
efd<-read.table("default of credit card clients.csv",sep=",",skip=2)
# ignore id
efd<-efd[,c(-1)]
x<-efd[,c(-ncol(efd))]
y<-efd[,ncol(efd)]
# train test split
n<-nrow(x)
split<-0.8
train<-sample(1:n, round(n*split))
xTrain<-x[train,]
xTest<-x[-train,]
yTrain<-y[train]
yTest<-y[-train]
# logistic regression with variety of regularization schemes
# ridge regression
ridge<-cv.glmnet(as.matrix(xTrain), yTrain, family="binomial", type.measure="class", alpha=0)
plot(ridge,main='ridge logistic regression')
# lasso regression
lasso<-cv.glmnet(as.matrix(xTrain), yTrain, family="binomial", type.measure="class", alpha=1)
plot(lasso,main='lasso logistic regression')
# elastic net regression
elastic1<-cv.glmnet(as.matrix(xTrain), yTrain, family="binomial", type.measure="class", alpha=0.3)
plot(elastic1,main='alpha=0.3 logistic regression')
elastic2<-cv.glmnet(as.matrix(xTrain), yTrain, family="binomial", type.measure="class", alpha=0.5)
plot(elastic2,main='alpha=0.5 logistic regression')
elastic3<-cv.glmnet(as.matrix(xTrain), yTrain, family="binomial", type.measure="class", alpha=0.7)
plot(elastic3,main='alpha=0.7 logistic regression')
# Test regression models after finding best regularization constant for each
ridgelambda<-ridge$lambda.min
lassolambda<-lasso$lambda.min
elasticnet1lambda<-elastic1$lambda.min
elasticnet2lambda<-elastic2$lambda.min
elasticnet3lambda<-elastic3$lambda.min
yhatlasso<-predict(lasso, as.matrix(xTest), s=lassolambda, type="class")
yhatridge<-predict(ridge, as.matrix(xTest), s=ridgelambda, type="class")
yhatelastic1<-predict(elastic1, as.matrix(xTest), s=elasticnet1lambda, type="class")
yhatelastic2<-predict(elastic2, as.matrix(xTest), s=elasticnet2lambda, type="class")
yhatelastic3<-predict(elastic3, as.matrix(xTest), s=elasticnet3lambda, type="class")
sum(yhatlasso==yTest)/nrow(xTest)
sum(yhatridge==yTest)/nrow(xTest)
sum(yhatelastic1==yTest)/nrow(xTest)
sum(yhatelastic2==yTest)/nrow(xTest)
sum(yhatelastic3==yTest)/nrow(xTest)