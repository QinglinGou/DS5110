import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
data_diabetes=datasets.load_diabetes()
data =  data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']
X, y = diabetes.data, diabetes.target
feature_names = np.array(diabetes.feature_names)
df =  pd.DataFrame(data,columns = feature_names)

# Create empty list to store results
rsquared = []

# Loop through features and calculate R-squared
for feature in df.columns:
  lr = LinearRegression()
  lr.fit(df[[feature]], y)
  rsquared.append(lr.score(df[[feature]], y))

# Zip features and results together
results = dict(zip(df.columns, rsquared))



b=[]
for i in range(1,10):
  sfs_selector = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select = i, direction ='forward')
  sfs_selector.fit(X, y)
  f=feature_names[sfs_selector.get_support()]
  b.append(f)

x = [list(i) for i in b]
x

#creating empty director that store feature and number of each feature
features_dict={}

#storing number of feature in features_dict
for i in range(len(x)):
	for j in range(len(x[i])):
		if x[i][j] not in features_dict:
			features_dict[x[i][j]]=0
		features_dict[x[i][j]]+=1
#copying features in result
result=[]
for key,value in features_dict.items():
	result.append(key)
#print results
res =result+['s6']
print("----The rank Of Q2 is:-----")
print(res)

new_rsq=[]
for feature in res:
  print("{}: {}".format(feature, results[feature]))
  new_rsq.append(results[feature])

plt.bar(res, new_rsq, color ='green',
        width = 0.5)
plt.xlabel('Features name')
plt.ylabel('Squared correlation(R squared)')
plt.title('Forward sequential feature selection ')
plt.show()
