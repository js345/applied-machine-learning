from __future__ import division
from dataLoader import readTrainingData, readValidationData, readMatchingData, readEvalData, readEvalResult
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import linearSVM
#import svm
import numpy as np
import naiveBayes
import randomForests
import knn

def calculateAccuracy(prediction,label):
    correct = 0
    for i in range(len(prediction)):
        if i == 10000:
            print correct / 10000
        if int(prediction[i]) == int(label[i]):
            correct += 1
    return correct / len(prediction)

def featureAdd(m):
    count = 0
    f = []
    for i in range(50):
        f.append((m[i]*m[i+73]))
        if m[i]*m[i+73] < 0:
            count += 1
    return np.array(f)

def writePrediction(prediction):
    f = open("eval_predict.txt",'w+')
    f.write("Id,Prediction\n")
    for i in range(len(prediction)):
        s = str(i)+","+str(prediction[i])+"\n"
        f.write(s)
    f.close()

def predict():
    testX = readEvalData().as_matrix()
    testY = readEvalResult().as_matrix()[:,1]
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]

    addedTrain = np.apply_along_axis(featureAdd,1,trainX)
    addedTest = np.apply_along_axis(featureAdd,1,testX)
    #trainX = np.absolute(trainX[:,:73] - trainX[:,73:])
    #testX = np.absolute(testX[:,:73] - testX[:,73:])
    trainX = np.concatenate((trainX,addedTrain),axis=1)
    testX = np.concatenate((testX,addedTest),axis=1)
    #trainX = np.absolute(trainX[:,:73] - trainX[:,73:])
    #testX = np.absolute(testX[:,:73] - testX[:,73:])
    #trainX = trainX[:,:9]
    #testX = testX[:,:9]
    #pca = PCA(n_components=3)
    #pca.fit(trainX)
    #trainX = pca.transform(trainX)
    mean = trainX.mean(axis=0)
    std = trainX.std(axis=0)
    trainX = (trainX - mean) / std
    testX = (testX - mean) / std
    model = svm.SVC(C=1.0)
    #model = RandomForestClassifier(n_estimators=30,criterion="entropy",max_depth=13,oob_score=True,min_samples_leaf=3)
    model.fit(trainX, trainY)
    #testX = pca.transform(testX)
    prediction = model.predict(testX)
    prediction = prediction.astype(int)
    print calculateAccuracy(prediction, testY)
    writePrediction(prediction)

if __name__ == '__main__':
    predict()