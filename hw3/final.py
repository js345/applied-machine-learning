from __future__ import division
from dataLoader import readTrainingData, readValidationData, readMatchingData, readEvalData, readEvalResult
from dataWriter import writePrediction
from sklearn import svm
from sklearn import preprocessing
import numpy as np

class finalClassifier():

    def __init__(self,trainX,trainY,testX,testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.normD = 2  # using 2 norms - a parameter to change
        self.normalization = preprocessing.StandardScaler().fit(trainX)    # normalization function - parameter to change

    def train(self):
        # classifier - look at sklearn svc or nusvc and tune parameters
        self.model = svm.SVC(C=1.0,gamma='auto')
        self.model.fit(self.trainX,self.trainY)

    def calculateAccuracy(self,prediction,label):
        correct = 0
        for i in range(len(prediction)):
            if i == 10000:
                print correct / 10000
            if int(prediction[i]) == int(label[i]):
                correct += 1
        return correct / len(prediction)

    def featureAdd(self,row):
        product = []
        sum = 0.0
        norm = 0.0
        for i in range(73):
            product.append((row[i]*row[i+73]))
            sum += row[i]*row[i+73]*self.featureWeight[i]*self.featureWeight[i+73]
            norm += pow(row[i]*self.featureWeight[i],self.normD) + pow(row[i+73]*self.featureWeight[i+73],self.normD)
        product.append(sum)
        product.append(pow(norm,1.0/self.normD))
        return np.array(product)

    def preprocess(self,featureWeight,dotWeight,cosWeight):
        # weight attributes # adjust them to preprocess and train
        self.dotWeight = dotWeight
        self.cosWeight = cosWeight
        self.featureWeight = np.tile(featureWeight,2)
        self.featureWeight = np.append(self.featureWeight,[self.dotWeight,self.cosWeight])
        # computing products aibi and dot product and cos theta using 2 norm
        addedTrain = np.apply_along_axis(self.featureAdd,1,self.trainX)
        addedTest = np.apply_along_axis(self.featureAdd,1,self.testX)
        # computing abs difference of every feature
        self.trainX = np.absolute(self.trainX[:,:73] - self.trainX[:,73:])
        self.testX = np.absolute(self.testX[:,:73] - self.testX[:,73:])
        # put together
        self.trainX = np.concatenate((self.trainX,addedTrain),axis=1)
        self.testX = np.concatenate((self.testX,addedTest),axis=1)
        # scale them - can change the sequence of operation to decide which to scale
        self.trainX = self.normalization.transform(trainX)
        self.testX = self.normalization.transform(testX)

    def predict(self):
        prediction = self.model.predict(self.testX)
        self.calculateAccuracy(prediction,self.testY)
        return prediction

if __name__ == '__main__':
    testX = readEvalData().as_matrix()
    testY = readEvalResult().as_matrix()[:,1]
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]
    mySVM = finalClassifier(trainX,trainY,testX,testY)
    # here to change attribute weights -- very important
    mySVM.preprocess(np.ones(73),1,1)
    mySVM.train()
    writePrediction(mySVM.predict())