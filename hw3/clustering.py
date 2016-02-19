from __future__ import division
from sklearn import svm
from sklearn.cluster import KMeans
from dataLoader import readTrainingData, readValidationData,readEvalData,readEvalResult
from dataWriter import writePrediction
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def calculateAccuracy(prediction,label):
    correct = 0
    for i in range(len(prediction)):
        if int(prediction[i]) == int(label[i]):
            correct += 1
    return correct / len(prediction)

def clustering(train,k) :
    trainX = train[:,1:]
    clusters = KMeans(n_clusters=k)
    clusters.fit(trainX)
    ypred = clusters.predict(trainX)
    cluster_trainX = list()
    for i in range(k):
        indexi = ypred == i
        cluster_trainX.append(train[indexi])

    model = list()

    for i in range(k):
        model.append(svm.SVC())
        model[i].fit(cluster_trainX[i][:,1:], cluster_trainX[i][:,0])


        
    return clusters, model

def predict(eval, clusters, model) :
    labels = list()
    for i in range(len(eval)) :
        k = clusters.predict(eval[i])
        tmp = np.array(eval[i]).reshape(1,-1)
        label = model[k].predict(tmp)
        labels.append(label)
    return labels

if __name__ == '__main__':
    train = readTrainingData().as_matrix()
    clusters,model = clustering(train, 3)
    evalData = readEvalData().as_matrix()
    labels = predict(evalData, clusters, model)
    result = readEvalResult().as_matrix()[:,1]
    acc = calculateAccuracy(labels,result)
    print acc
    #writePrediction(labels)



