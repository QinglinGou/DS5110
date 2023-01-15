import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target


pca = PCA() # Default n_components = min(n_samples, n_features)
X_train_pc = pca.fit_transform(X)
pd.DataFrame(pca.components_.T).loc[:4,:]

lin_reg = LinearRegression()
rmse_list = []
cv = KFold(n_splits=10, shuffle=True, random_state=42)
for i in range(1, X_train_pc.shape[1]+1):
    rmse_score = -1*  cross_val_score(lin_reg,
                       X_train_pc[:,:i], # Use first k principal components
                       y,
                       cv=cv,
                       scoring='neg_root_mean_squared_error').mean()
    rmse_list.append(rmse_score)

lin_reg = LinearRegression().fit(X, y)
lr_score_train = -1 * cross_val_score(lin_reg, X, y, cv=cv, scoring='neg_root_mean_squared_error').mean()

plt.plot(rmse_list, '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('RMSE')
plt.title('Quality')
plt.xlim(xmin=-1);
plt.xticks(np.arange(X_train_pc.shape[1]), np.arange(1, X_train_pc.shape[1]+1))
plt.axhline(y=lr_score_train, color='g', linestyle='-')
plt.show()