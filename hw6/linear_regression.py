from __future__ import division
from sklearn import linear_model
from scipy import stats
from scipy.special import boxcox, inv_boxcox
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("./Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt",delimiter=",",skiprows=0)
num_row,num_feature = data.shape[0],data.shape[1]-2
x = data[:,:num_feature]
y = data[:,num_feature:]
latitude = y[:,0]
longitude = y[:,1]
# add a column of ones to x
# x = np.append(x,np.ones((num_row,1)),axis=1)# Create linear regression object
regr = linear_model.LinearRegression()
def regress(x,y,y_label):
    regr.fit(x,y)
    print "R squared: " + str(regr.score(x,y))
    # Plot outputs
    fig = plt.figure()
    plt.scatter(y, regr.predict(x), color='blue')
    plt.xlabel(y_label)
    plt.ylabel('predicted')

regress(x,latitude,'latitude')

regress(x,longitude,'longitude')

def boxcox(x,y,y_label):
    box_cox, maxlog = stats.boxcox(y + abs(min(y)) + 1)
    regr.fit(x,box_cox)
    box_cox_predict = regr.predict(x)
    y_predict = inv_boxcox(box_cox_predict,maxlog) - abs(min(y)) - 1
    print "R squared: " + str(np.var(y_predict)/np.var(y))
    # Plot outputs
    fig = plt.figure()
    plt.scatter(y, y_predict, color='blue')
    plt.xlabel(y_label)
    plt.ylabel('predicted')


boxcox(x,latitude,'latitude')


boxcox(x,longitude,'longitude')