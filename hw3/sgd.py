from __future__ import division
from sklearn import linear_model
from dataLoader import readTrainingData, readValidationData, readEvalData, readEvalResult
from dataWriter import featureAdd, writePrediction
from sklearn.svm import NuSVC
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import naiveBayes

def sgd(trainX, trainY, testX, testY):
    #clf = linear_model.SGDClassifier(loss='log')
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(trainX, trainY)
    prediction = clf.predict(testX)
    return prediction.astype(int)

if __name__ == '__main__':
    testX = readEvalData().as_matrix()
    testY = readEvalResult().as_matrix()[:,1]
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]
    #addedTrain = np.apply_along_axis(featureAdd,1,trainX)
    #addedTest = np.apply_along_axis(featureAdd,1,testX)
    #trainX = np.concatenate((trainX,addedTrain.reshape(addedTrain.size,1)),axis=1)
    #testX = np.concatenate((testX,addedTest.reshape(addedTest.size,1)),axis=1)
    mean = trainX.mean(axis=0)
    std = trainX.std(axis=0)
    trainX = (trainX - mean) / std
    testX = (testX - mean) / std
    prediction = sgd(trainX, trainY, testX, testY)
    print naiveBayes.calculateAccuracy(prediction,testY)
    #writePrediction(prediction)