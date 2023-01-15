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
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X = digits.data
y = digits.target
target_names=["0","1","2","3","4","5","6","7","8","9"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)



model = RandomForestClassifier()
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
param_range = np.arange(1, 100, 2)
train_scores, test_scores = validation_curve(
    model, ##
    X, y,param_name='n_estimators', param_range=param_range,
    cv=10,scoring="accuracy", n_jobs=1 )


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_scores.mean(axis=1), color="darkorange",label="Training error")
plt.plot(param_range, test_scores.mean(axis=1), color="navy",label="Testing error")
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=2,
)
plt.legend()

plt.xlabel("n_estimators of Random Forest")
plt.ylabel("accuracy")
_ = plt.title("Validation curve for Random Forest")
plt.savefig("figs/Validation curve for Random Forest.png")
plt.show()


from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': np.arange(1, 100,5)
}

grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy',n_jobs=1)
grid_search = grid_search.fit(Xtrain, ytrain)
best_accuracy= grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))


digits = load_digits()
X = digits.data
y = digits.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

a=int(str(best_parameters)[16:-1])
model = RandomForestClassifier(n_estimators=a)
model.fit(Xtrain, ytrain)
yfit = model.predict(Xtest)
from sklearn.metrics import classification_report
target_names=["0","1","2","3","4","5","6","7","8","9"]
print(classification_report(ytest, yfit,target_names=target_names))

print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, yfit)))


mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
