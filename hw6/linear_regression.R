library(glmnet)
efd<-as.matrix(read.table('./Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt',sep=",",header=FALSE))
x<-efd[,-c(ncol(efd)-1,ncol(efd))]
latitude<-efd[,ncol(efd)-1]
longitude<-efd[,ncol(efd)]
# train test split
n<-nrow(x)
split<-0.8
train<-sample(1:n, round(n*split))
xTrain<-x[train,]
xTest<-x[-train,]
# x vs latitude
y<-latitude
yTrain<-y[train]
yTest<-y[-train]
ols<-lm(yTrain ~ xTrain)
ridge<-cv.glmnet(xTrain,yTrain,alpha=0)
lasso<-cv.glmnet(xTrain,yTrain,alpha=1)
elasticnet1<-cv.glmnet(xTrain, yTrain, alpha=0.3)
elasticnet2<-cv.glmnet(xTrain, yTrain, alpha=0.5)
elasticnet3<-cv.glmnet(xTrain, yTrain, alpha=0.7)
ridgelambda<-ridge$lambda.min
lassolambda<-lasso$lambda.min
elasticnet1lambda<-elasticnet1$lambda.min
elasticnet2lambda<-elasticnet2$lambda.min
elasticnet3lambda<-elasticnet3$lambda.min
ridgelambda
lassolambda
elasticnet1lambda
elasticnet2lambda
elasticnet3lambda
plot(ridge, main="ridge regression vs laitude")
plot(lasso, main="lasso regression vs laitude")
plot(elasticnet1, main="elastic regression vs laitude (alpha=0.3)")
plot(elasticnet2, main="elastic regression vs laitude (alpha=0.5)")
plot(elasticnet3, main="elastic regression vs laitude (alpha=0.7)")
# Test regression models after finding best regularization constant for each
yhatlasso<-predict(lasso, xTest, s=lassolambda)
yhatridge<-predict(ridge, xTest, s=ridgelambda)
yhatelastic1<-predict(elasticnet1, xTest, s=elasticnet1lambda)
yhatelastic2<-predict(elasticnet2, xTest, s=elasticnet2lambda)
yhatelastic3<-predict(elasticnet3, xTest, s=elasticnet3lambda)
# Compare MSE on test data
sum((yTest - yhatlasso)^2) / nrow(xTest)
sum((yTest - yhatridge)^2) / nrow(xTest)
sum((yTest - yhatelastic1)^2) / nrow(xTest)
sum((yTest - yhatelastic2)^2) / nrow(xTest)
sum((yTest - yhatelastic3)^2) / nrow(xTest)

# x vs longitude
y<-longitude
yTrain<-y[train]
yTest<-y[-train]
ols<-lm(yTrain ~ xTrain)
ridge<-cv.glmnet(xTrain,yTrain,alpha=0)
lasso<-cv.glmnet(xTrain,yTrain,alpha=1)
elasticnet1<-cv.glmnet(xTrain, yTrain, alpha=0.3)
elasticnet2<-cv.glmnet(xTrain, yTrain, alpha=0.5)
elasticnet3<-cv.glmnet(xTrain, yTrain, alpha=0.7)
ridgelambda<-ridge$lambda.min
lassolambda<-lasso$lambda.min
elasticnet1lambda<-elasticnet1$lambda.min
elasticnet2lambda<-elasticnet2$lambda.min
elasticnet3lambda<-elasticnet3$lambda.min
ridgelambda
lassolambda
elasticnet1lambda
elasticnet2lambda
elasticnet3lambda
plot(ridge, main="ridge regression vs longitude")
plot(lasso, main="lasso regression vs longitude")
plot(elasticnet1, main="elastic regression vs longitude (alpha=0.3)")
plot(elasticnet2, main="elastic regression vs longitude (alpha=0.5)")
plot(elasticnet3, main="elastic regression vs longitude (alpha=0.7)")
# Test regression models after finding best regularization constant for each
yhatlasso<-predict(lasso, xTest, s=lassolambda)
yhatridge<-predict(ridge, xTest, s=ridgelambda)
yhatelastic1<-predict(elasticnet1, xTest, s=elasticnet1lambda)
yhatelastic2<-predict(elasticnet2, xTest, s=elasticnet2lambda)
yhatelastic3<-predict(elasticnet3, xTest, s=elasticnet3lambda)
# Compare MSE on test data
sum((yTest - yhatlasso)^2) / nrow(xTest)
sum((yTest - yhatridge)^2) / nrow(xTest)
sum((yTest - yhatelastic1)^2) / nrow(xTest)
sum((yTest - yhatelastic2)^2) / nrow(xTest)
sum((yTest - yhatelastic3)^2) / nrow(xTest)

