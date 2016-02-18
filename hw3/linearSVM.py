from __future__ import division
from sklearn import svm
from dataLoader import readTrainingData, readValidationData

def calculateAccuracy(prediction,label):
    correct = 0
    for i in range(len(prediction)):
        if int(prediction[i]) == int(label[i]):
            correct += 1
    return correct / len(prediction)

def applyLinearSVM():
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]
    model = svm.LinearSVC()
    model.fit(trainX, trainY)
    prediction = model.predict(trainX)
    print "training acc"
    print calculateAccuracy(prediction, trainY)
    return model

def testModel(number,model):
    feature,label = readValidationData(number)
    testX = feature.as_matrix()
    testY = label.as_matrix()
    prediction = model.predict(testX)
    return calculateAccuracy(prediction, testY)

if __name__ == '__main__':
    model = applyLinearSVM()
    for i in range(1,4):
        accuracy = testModel(i,model)
        print accuracy