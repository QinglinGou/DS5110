import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
parameters = {'base_estimator__max_depth':np.arange(1, 20)}

grid_search = GridSearchCV(abc, parameters,scoring='accuracy',n_jobs=1,return_train_score=True)
cv = grid_search.fit(Xtrain,ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))

test_scores = cv.cv_results_['mean_test_score']
train_scores = cv.cv_results_['mean_train_score']
train_scores_std = cv.cv_results_['std_train_score']
test_scores_std =cv.cv_results_['std_test_score']
param_range= np.arange(1, 20)
plt.plot(param_range,test_scores, label='test')
plt.fill_between(
    param_range,
    test_scores - test_scores_std,
    test_scores + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=0.1,
)
plt.plot(param_range,train_scores, label='train')
plt.legend(loc='best')
_ = plt.title("Validation curve for adaboost in max_depth")
plt.savefig("figs/validation curve of adaboost in max_depth")
plt.show()




a=best_parameters.get('base_estimator__max_depth')
ab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
from sklearn.model_selection import GridSearchCV
parameters = {
    'base_estimator__max_depth':[a],
    'n_estimators': [1, 10, 20, 30,50,100,150,200]
}
grid_search = GridSearchCV(ab_clf, parameters, cv=5, scoring='accuracy',n_jobs=1,return_train_score=True)
cv = grid_search.fit(Xtrain, ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))

#####

test_scores = cv.cv_results_['mean_test_score']
train_scores = cv.cv_results_['mean_train_score']
train_scores_std = cv.cv_results_['std_train_score']
test_scores_std =cv.cv_results_['std_test_score']
param_range= [1, 10, 20, 30,50,100,150,200]
plt.plot(param_range,test_scores, label='test')
plt.fill_between(
    param_range,
    test_scores - test_scores_std,
    test_scores + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=0.1,
)
plt.plot(param_range,train_scores, label='train')
plt.legend(loc='best')
_ = plt.title("Validation curve for adaboost in n_estimators")
plt.savefig("figs/validation curve of adaboost in n_estimators")
plt.show()

####








ab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
b = best_parameters.get('n_estimators')
from sklearn.model_selection import GridSearchCV
parameters = {
    'base_estimator__max_depth':[a],
    'n_estimators': [b],
    "learning_rate":np.arange(0.5, 1.5, 0.1)
}

grid_search = GridSearchCV(ab_clf, parameters, cv=5, scoring='accuracy',n_jobs=1,return_train_score=True)
cv = grid_search.fit(Xtrain, ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))


#####
test_scores = cv.cv_results_['mean_test_score']
train_scores = cv.cv_results_['mean_train_score']
train_scores_std = cv.cv_results_['std_train_score']
test_scores_std =cv.cv_results_['std_test_score']
param_range= np.arange(0.5, 1.5, 0.1)
plt.plot(param_range,test_scores, label='test')
plt.fill_between(
    param_range,
    test_scores - test_scores_std,
    test_scores + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=0.1,
)
plt.plot(param_range,train_scores, label='train')
plt.legend(loc='best')
_ = plt.title("Validation curve for adaboost in learning rate")
plt.savefig("figs/validation curve of adaboost in learning rate")
plt.show()








ab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
c=best_parameters.get('learning_rate')
from sklearn.model_selection import GridSearchCV
parameters = {
    'base_estimator__max_depth':[a],
    'n_estimators': [b],
    "learning_rate":[c],
    "algorithm": ["SAMME","SAMME.R"]
}

grid_search = GridSearchCV(ab_clf, parameters, cv=10, scoring='accuracy',n_jobs=1,return_train_score=True)
grid_search = grid_search.fit(Xtrain, ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))



dt_stump = DecisionTreeClassifier(max_depth=a, min_samples_leaf=1)
d=best_parameters.get("algorithm")
model = AdaBoostClassifier(base_estimator=dt_stump,n_estimators = b, learning_rate=c,algorithm=d )
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
