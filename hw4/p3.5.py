__author__ = 'weixin1'
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression as PLS
import numpy as np

df = pd.read_csv("wine.data.txt", sep=",", skipinitialspace=True, skiprows=0, header=None).as_matrix()
x = df[:,1:]
y = df[:,0]
covr = np.cov(x.T)
w,v = np.linalg.eig(covr)
w,v = zip(*sorted(zip(w, v)))
print w , v

#plt.hlines(1,1,20)
#plt.eventplot(w1, orientation= 'horizontal', colors = 'b')
#plt.axis('off')
#plt.show()