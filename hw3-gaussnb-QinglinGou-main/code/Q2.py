# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class GaussianNB(object):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    def GaussianProbability(self,x, mean, var):
        return np.array([np.exp(-np.power(x[i] - mean[i], 2) / (2 * np.power(var[i], 2))) for i in range(len(x))])
    def fit(self):
        self.weight = np.array([[np.mean(self.data[self.labels == label],axis=0),np.var(self.data[self.labels == label],axis=0)] for label in np.unique(self.labels)])
    def predict(self,samples):
        return np.array([np.unique(self.labels)[np.argmax([self.GaussianProbability(sample,self.weight[:,0,:][i],self.weight[:,1,:][i]).prod()  for i in range(len(np.unique(self.labels)))])] for sample in samples])

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
Gau = GaussianNB(X,y)
Gau.fit()

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = Gau.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

plt.show()
