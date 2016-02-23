from __future__ import division
from dataLoader import readTrainingData, readValidationData, readMatchingData, readEvalData, readEvalResult
from dataWriter import writePrediction
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import time


class finalClassifier():
    best_score = 0.79685

    def __init__(self,trainX,trainY,testX,testY):
        self.number_of_split = 1
        self.range_of_split = len(trainY) / self.number_of_split
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.normD = 2  # using 2 norms - a parameter to change
        #self.normalization = preprocessing.StandardScaler().fit(trainX)    # normalization function - parameter to change

    def train(self):
        self.model = list()
        # split data into smaller subsets and then voting
        for i in range(self.number_of_split):
            # classifier - look at sklearn svc or nusvc and tune parameters
            self.model.append(svm.SVC(C=1.1,gamma='auto'))
            self.model[i].fit(self.trainX[i*self.range_of_split:(i+1)*self.range_of_split],self.trainY[i*self.range_of_split:(i+1)*self.range_of_split])

    def calculateAccuracy(self,prediction,label):
        correct = 0
        for i in range(len(prediction)):
            if i == 10000:
                print correct / 10000
            if int(prediction[i]) == int(label[i]):
                correct += 1
        print correct / len(prediction)
        return correct / len(prediction)

    def featureAdd(self,row):
        product = []
        squares = []
        ab = []
        sum = 0.0
        norm = 0.0
        for i in range(73):
            if self.featureWeight[i] != 0:
                p = row[i]*row[i+73]
                diff = row[i] - row[i+73]
                ab.append(abs(diff))
                product.append(p)
                squares.append(pow(p,self.normD))
                sum += p
                norm += pow(row[i],self.normD) + pow(row[i+73],self.normD)

        squares.append(sum)
        squares.append(pow(norm,1.0/self.normD))
        product = np.concatenate((np.array(ab),np.array(product)),axis=0)
        return np.concatenate((np.array(product),np.array(squares)),axis=0)

    def preprocess(self,featureWeight,dotWeight,cosWeight):
        # weight attributes # adjust them to preprocess and train
        self.dotWeight = dotWeight
        self.cosWeight = cosWeight
        self.featureWeight = np.tile(featureWeight,2)
        self.featureWeight = np.append(self.featureWeight,[self.dotWeight,self.cosWeight])
        # computing products aibi and abs diff
        addedTrain = np.apply_along_axis(self.featureAdd,1,self.trainX)
        addedTest = np.apply_along_axis(self.featureAdd,1,self.testX)

        # put together
        self.trainX = np.concatenate((self.trainX,addedTrain),axis=1)
        self.testX = np.concatenate((self.testX,addedTest),axis=1)

        # set up mask
        mask = []
        for i in range(len(self.featureWeight)):
            if self.featureWeight[i] == 0:
                mask.append(i)
        self.trainX = np.delete(self.trainX,mask,axis=1)
        self.testX = np.delete(self.testX,mask,axis=1)

        # scale them - can change the sequence of operation to decide which to scale
        mean = self.trainX.mean(axis=0)
        std = self.trainX.std(axis=0)
        self.trainX = (self.trainX - mean) / std
        self.testX = (self.testX - mean) / std

    def predict(self):
        prediction = np.zeros(len(self.testY))
        for i in range(self.number_of_split):
            prediction += self.model[i].predict(self.testX).astype(int)
        prediction /= self.number_of_split
        prediction = np.rint(prediction)
        self.calculateAccuracy(prediction,self.testY)
        return prediction

if __name__ == '__main__':
    start = time.time()
    testX = readEvalData().as_matrix()
    testY = readEvalResult().as_matrix()[:,1]
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]
    mySVM = finalClassifier(trainX,trainY,testX,testY)
    # here to change attribute weights -- very important
    featureWeight = np.ones(73)
    featureWeight[19] = 0 # smile
    featureWeight[22] = 0 # blurry
    featureWeight[23] = 0 # harsh lighting
    featureWeight[24] = 0 # flash
    featureWeight[25] = 0 # soft lighting
    featureWeight[26] = 0 # outdoor
    featureWeight[55] = 0 # color photo
    featureWeight[56] = 0 # pose photo
    mySVM.preprocess(featureWeight,1,1)
    mySVM.train()
    answer = mySVM.predict()
    writePrediction(answer)
    print("--- %s seconds ---" % (time.time() - start))