import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn import datasets
diabetes = load_diabetes()
data_diabetes=datasets.load_diabetes()
data =  data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']
X, y = diabetes.data, diabetes.target
feature_names = np.array(diabetes.feature_names)
df =  pd.DataFrame(data,columns = feature_names)
X, y = datasets.load_diabetes(return_X_y=True)

alpha,index,coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)


new_index=[]
for i in range(len(index)):
  s = feature_names[int(index[i])]
  new_index.append(s)

from sys import platform
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
plt.legend(feature_names,bbox_to_anchor=(1, 1),ncol=1)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle="dashed")
plt.xlabel("|coef| / max|coef|")
plt.ylabel("Coefficients")
plt.title("LASSO Path")
plt.axis("tight")
plt.show()

print(f"the order of lasso path:",new_index)