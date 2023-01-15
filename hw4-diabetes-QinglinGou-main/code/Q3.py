import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import  cross_val_score
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
data=pd.DataFrame(df[res[3:10]])

cov=[]
for i in data.columns:
  model = LinearRegression()
  top4 = pd.merge(df.loc[0:,['bmi', 's5', 'bp']],df[i], left_index=True, right_index=True)
  model.fit(top4,y)
  cov.append(cross_val_score(model,top4,y).mean())

subset =dict(zip(data, cov))
print(subset)
plt.ylim(0.45, 0.475)
plt.bar(res[3:10], cov, color ='green',
        width = 0.5)
plt.xlabel('Number 4 Feature name')
plt.ylabel('score')
plt.title('  ')
plt.show()
