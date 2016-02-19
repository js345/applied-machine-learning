from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dataLoader import readTrainingData, readValidationData\


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
'''
train = readTrainingData().as_matrix()
np.random.shuffle(train)
train = train[:5000]
pca = PCA(n_components=3)
trainF = train[:,1:]

trainF = np.absolute(trainF[:,:73] - trainF[:,73:])
mean = trainF.mean(axis=0)
std = trainF.std(axis=0)
trainX = (trainF - mean) / std
pca.fit(trainF)
X = pca.transform(trainF)

trainX = train[:,1]
trainY = train[:,2]
trainZ = train[:,3]
trainLabel = train[:,0]

label0 = trainLabel==0
label1 = trainLabel==1
X0 = X[label0]
X1 = X[label1]
trainX0 = trainX[label0]
trainX1 = trainX[label1]
trainY0 = trainY[label0]
trainY1 = trainY[label1]
trainZ0 = trainZ[label0]
trainZ1 = trainZ[label1]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X0[:,0],X0[:,1],c = 'r', marker = 'o')
ax.scatter(X1[:,0],X1[:,1],c = 'b', marker = '^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()