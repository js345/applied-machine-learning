from __future__ import division
from sklearn import svm
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

# read data
data = list()
with open('adult.data', 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    for row in csvreader:
        if row == []:
            break
        if row[0] == "?" or row[2] == "?" or row[4] == "?" or row[10] == "?" or row[11] == "?" or row[12] == "?" or row[14] == "?":
            continue
        if row[14] == ">50K":
            row[14] = 1
        else:
            row[14] = -1
        example = [float(row[0]),float(row[2]),float(row[4]),float(row[10]),float(row[11]),float(row[12]),row[14]]
        data.append(example)

# z-score normalization
data = np.array(data)
label = data[:,-1:]
feature = data[:,:-1]
mean = feature.mean(axis=0)
std = feature.std(axis=0)
feature = (feature - mean) / std
data = np.append(feature,label,axis=1)


# split data into 10% validation 10% test 80% train
testX = list()
testY = list()
validationX = list()
validationY = list()
train = list()

for row in data:
    p = random.random()
    if p < 0.1:
        validationX.append(row[0:-1])
        validationY.append(row[-1])
    elif p < 0.2:
        testX.append(row[0:-1])
        testY.append(row[-1])
    else:
        train.append(row)

testX = np.array(testX)
testY = np.array(testY)
validationX = np.array(validationX)
validationY = np.array(validationY)
train = np.array(train)
# svm main loop
regWeight = [0.001, 0.01, 0.1, 1]
epochs = 50
steps = 300
# step length 1/ a*epoch + b
a = 0.01
b = 50
bestlmbda = 0
bestAccuracy = 0
bestA = 0
bestB = 0

def calculateAccuracy(Xs, Ys, A, B):
    labeledRight = 0
    for i in range(len(Ys)):
        yLabel = A.dot(Xs[i]) + B
        if (yLabel < 0) ^ (Ys[i] >= 0):
            labeledRight += 1
    return labeledRight / len(Ys)

for lmbda in regWeight:
    A = np.zeros(len(testX[0]))
    B = 0
    accuracyHolder = list()
    for i in range(epochs):
        stepLength = 1.0 / (a*i + b)
        np.random.shuffle(train)
        crossV = train[0:500]
        smallTrain = train[500:]
        trainX = smallTrain[:, :-1]
        trainY = smallTrain[:, -1]
        crossX = crossV[:,:-1]
        crossY = crossV[:,-1]
        accuracyHolder.append(calculateAccuracy(crossX, crossY, A, B))
        for j in range(steps):
            randomPoint = np.random.randint(0,len(trainY))
            xPoint = trainX[randomPoint]
            yPoint = trainY[randomPoint]
            predictLabel = xPoint.dot(A) + B
            if predictLabel * yPoint >= 1.0:
                gradientA = lmbda * A
                gradientB = 0
            else:
                gradientA = ((-yPoint)*xPoint) + lmbda * A
                gradientB = -yPoint
            A -= (stepLength * gradientA)
            B -= (stepLength * gradientB)

    accuracy = calculateAccuracy(validationX, validationY, A, B)
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestlmbda = lmbda
        bestA = A
        bestB = B

    plt.plot(accuracyHolder)
    plt.ylim(0,1)
    plt.show()

finalAccuracy = calculateAccuracy(testX, testY, bestA, bestB)
print "best accuracy "+ str(finalAccuracy)

"""
clf = svm.SVC()
clf.fit(trainX,trainY)
number = 0
predict = clf.predict(testX)
for i in range(len(predict)):
    if predict[i] == testY[i]:
        number += 1
print number / len(testY)"""