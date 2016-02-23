from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from dataLoader import readTrainingData, readValidationData, readEvalData, readEvalResult

def calculateAccuracy(prediction,label):
    correct = 0
    for i in range(len(prediction)):
        if int(prediction[i]) == int(label[i]):
            correct += 1
    return correct / len(prediction)

def applyRandomForest():
    train = readTrainingData().as_matrix()
    trainX = train[:,1:]
    trainY = train[:,0]
    model = RandomForestClassifier(n_estimators=30,criterion="entropy",max_depth=13,oob_score=True,min_samples_leaf=3)
    model.fit(trainX, trainY)
    prediction = model.predict(trainX)
    print "training acc: " + str(calculateAccuracy(prediction, trainY))
    return model

def testModel(number,model):
    feature,label = readValidationData(number)
    testX = feature.as_matrix()
    testY = label.as_matrix()
    prediction = model.predict(testX)
    return calculateAccuracy(prediction, testY)

def testEval():
    testX = readEvalData().as_matrix()
    testY = map(lambda x: x[1], readEvalResult().as_matrix())
    prediction = model.predict(testX)
    return calculateAccuracy(prediction, testY)

if __name__ == '__main__':
    print "Random Forests"
    model = applyRandomForest()
    for i in range(1,4):
        accuracy = testModel(i,model)
        print "validation" + str(i) + " acc: " + str(accuracy)

    print "eval acc: " + str(testEval())