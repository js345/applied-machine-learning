import pandas as pd


def readTrainingData():
    df = pd.read_csv("pubfig_dev_50000_pairs.txt", sep="\t", skipinitialspace=True, skiprows=1, header=None)
    return df

def readValidationData(number):
    path = "pubfig_kaggle_"+str(number)
    feature = pd.read_csv(path+".txt", sep="\t", skipinitialspace=True, skiprows=2, header=None)
    label = pd.read_csv(path+"_solution.txt", sep=",", skipinitialspace=True, skiprows=0)
    return feature,label["Prediction"]

def readMatchingData():
    df = pd.read_csv("pubfig_attributes.txt", sep="\t", skipinitialspace=True, skiprows=2, header=None)
    return df

def readEvalData():
    df = pd.read_csv("pubfig_kaggle_eval.txt", sep="\t", skipinitialspace=True, skiprows=2, header=None)
    return df

def readEvalResult():
    df = pd.read_csv("eval_solution.txt", sep=",", skipinitialspace=True, skiprows=1, header=None)
    return df

if __name__ == '__main__':
    print readEvalData()
