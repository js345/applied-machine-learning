from dataLoader import readTrainingData, readValidationData, readMatchingData, readEvalData, readEvalResult
from sklearn import svm
import linearSVM
#import svm
import naiveBayes
import randomForests
import knn

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
    model = svm.SVC()
    model.fit(trainX, trainY)
    prediction = model.predict(testX)
    prediction = prediction.astype(int)
    print naiveBayes.calculateAccuracy(prediction, testY)
    writePrediction(prediction)

if __name__ == '__main__':
    predict()