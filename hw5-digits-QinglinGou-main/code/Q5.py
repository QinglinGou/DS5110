from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt


digits = load_digits()
X = digits.data
y = digits.target
target_names=["0","1","2","3","4","5","6","7","8","9"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y)
svc = SVC(kernel='rbf')

# Tuning gamma
parameters = {'gamma':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0008, 0.001, 0.005]}
grid_search = GridSearchCV(svc, parameters,scoring='accuracy',n_jobs=1,return_train_score=True)
cv = grid_search.fit(Xtrain,ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))

test_scores = cv.cv_results_['mean_test_score']
train_scores = cv.cv_results_['mean_train_score']
train_scores_std = cv.cv_results_['std_train_score']
test_scores_std =cv.cv_results_['std_test_score']
param_range= [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0008, 0.001, 0.005]
plt.ylim(0.9, 1.02)
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
_ = plt.title("Validation curve for SVC in gamma")
plt.savefig("figs/validation curve of SVC in gamma")
plt.show()


#Tuning C
a=best_parameters.get('gamma')
svc = SVC(kernel='rbf')
from sklearn.model_selection import GridSearchCV
parameters = {
    'gamma':[a],
    'C': [1,2,3,4,5]
}
grid_search = GridSearchCV(svc, parameters, cv=5, scoring='accuracy',n_jobs=1,return_train_score=True)
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
param_range= [1,2,3,4,5]
plt.ylim(0.96, 1.01)
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
_ = plt.title("Validation curve for SVC in C")
plt.savefig("figs/validation curve of SVC in C")
plt.show()




# Tuning best
from sklearn.model_selection import GridSearchCV
parameters = {'C': [1,2,3,4,5],
              'gamma': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0008, 0.001, 0.005],
              'kernel': ['linear', 'poly', 'rbf']
              }

grid_search = GridSearchCV(estimator=SVC(),

                           param_grid=parameters,

                           scoring='accuracy',

                           cv=10,

                           n_jobs=1)

grid_search = grid_search.fit(Xtrain, ytrain)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('\nBest_accuracy: {:.2f}'.format(best_accuracy))
print('Best_parameters: {}\n'.format(best_parameters))

model = SVC(C=best_parameters.get("C"),kernel =best_parameters.get("kernel"), gamma = best_parameters.get("gamma") )
model.fit(Xtrain, ytrain)
yfit = model.predict(Xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,target_names=target_names))

print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, yfit)))


mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
