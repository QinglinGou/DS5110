import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn import tree


digits = load_digits()
X = digits.data
y = digits.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y)

model = GaussianNB()
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)

target_names=["0","1","2","3","4","5","6","7","8","9"]
print(classification_report(ytest, y_pred,target_names=target_names))

print('\nTrain accuracy(balanced): {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy(balanced): {:.2f}\n'.format(accuracy_score(ytest, y_pred)))


mat = confusion_matrix(ytest, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
