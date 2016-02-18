from __future__ import division
from dataLoader import readValidationData, readMatchingData,readEvalData
#from dataWriter import writePrediction
from pyflann import *
import numpy as np

def calculateAccuracy(prediction,label):
    correct = 0
    for i in range(len(prediction)):
        if int(prediction[i]) == int(label[i]):
            correct += 1
    print correct,len(prediction)
    return correct / len(prediction)

def approxKNN(number):
    matchData = readMatchingData().as_matrix()
    name = matchData[:,:2]
    attribute = matchData[:,2:]
    feature,label = readValidationData(number)
    testX = feature.as_matrix()
    firstFace,secondFace = testX[:,:73],testX[:,73:]
    testY = label.as_matrix()
    flann = FLANN()

    result1,dist1 = flann.nn(attribute.astype(float32),firstFace.astype(float32), num_neighbors=1)
    result2,dist2 = flann.nn(attribute.astype(float32),secondFace.astype(float32), num_neighbors=1)

    prediction = classify(result1,result2,name)

    return calculateAccuracy(prediction,testY)

def classify(result1,result2,name):
    assert len(result1) == len(result2)
    prediction = list()
    for i in range(len(result1)):
        if name[result1[i],0] == name[result2[i],0]:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)

def knnKaggle():
    matchData = readMatchingData().as_matrix()
    name = matchData[:,:2]
    attribute = matchData[:,2:]
    testX = readEvalData().as_matrix()
    firstFace,secondFace = testX[:,:73],testX[:,73:]
    flann = FLANN()
    result1,dist1 = flann.nn(attribute.astype(float32),firstFace.astype(float32), num_neighbors=1)
    result2,dist2 = flann.nn(attribute.astype(float32),secondFace.astype(float32), num_neighbors=1)
    return classify(result1,result2,name)

if __name__ == '__main__':

    for i in range(1,4):
        print approxKNN(i)

    #writePrediction(knnKaggle())