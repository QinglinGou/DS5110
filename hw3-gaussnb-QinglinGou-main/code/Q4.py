
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


import argparse

parser = argparse.ArgumentParser(description='Plot for make_blobs dataset')
parser.add_argument('cluster_std', metavar='std', type=float, default= 1.5,
                    help='The standard deviation of the clusters.')
args = parser.parse_args()
#print("Here's what your type:", args.cluster_std)

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=args.cluster_std)

Gau = GaussianNB(X,y)
Gau.fit()
fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)


xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 200)
yg = np.linspace(ylim[0], ylim[1], 200)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

Z = Gau.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) 
ax.contour(xx, yy, Z, [0.5], colors='red')  
ax.set(xlim=xlim, ylim=ylim)


def comp_conf(actual, predicted):   
    classe = np.unique(actual)
    conf = np.zeros((len(classe), len(classe)))  
    for i in range(len(classe)):
        for j in range(len(classe)):
           conf[i, j] = np.sum((actual == classe[i]) & (predicted == classe[j]))

    return conf

y_predict = Gau.predict(X)
print("---Confusion Matrix---")
print(comp_conf(y, y_predict))

conf_matrix= comp_conf(y, y_predict)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Greens, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()





