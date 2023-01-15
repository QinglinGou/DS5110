
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

data_diabetes=datasets.load_diabetes()
data =  data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']
df =  pd.DataFrame(data,columns = feature_names)
X=df
y = target

# Create empty list to store results
rsquared = []

# Loop through features and calculate R-squared
for feature in df.columns:
  lr = LinearRegression()
  lr.fit(df[[feature]], y)
  rsquared.append(lr.score(df[[feature]], y))

# Zip features and results together
results = dict(zip(df.columns, rsquared))

# Sort results
sorted_results = sorted(results, key=lambda x: results[x], reverse=True)
sorted_results
# Print sorted results
print("----The rank Of Q1 is:-----")
rsq=[]
for feature in sorted_results:
  print("{}: {}".format(feature, results[feature]))
  rsq.append(results[feature])

plt.bar(sorted_results, rsq, color ='green',
        width = 0.5)
plt.xlabel('Features name')
plt.ylabel('Squared correlation(R squared)')
plt.title('univariate regression')
plt.show()
