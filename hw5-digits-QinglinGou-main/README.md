# hw5-digits-QinglinGou
Learning goals:

* Practice with scikit-learn API using the digits dataset
* Comparison of various classification methods
* Hyperparameter tuning and model selection


# Question 1
Use the [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) to investigate performance by class. In particular, note the variation in the "support" column. Propose a solution if you think the support may be problematic. Implement your proposed solution and comment on the results.
```
make Q1
```


The problem is that in the dataset splitting, test dataset may unbalanced.
Solution:
`stratify` parameter will preserve the proportion of target as in original dataset, in the train and test datasets as well.

code: `Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y)`

```
Train accuracy(balanced): 0.86
Test accuracy(balanced): 0.84
```





# Question 2
Use a decision tree classifier with the same data. Investigate model performance using a validation curve. Comment briefly on the results (including comparison with results above).
```
make Q2
```


<img src="figs/Validation curve for decision tree.png" width="500">

```
Best_accuracy: 0.86
Best_parameters: {'max_depth': 10}
```

```
Train accuracy: 0.99
Test accuracy: 0.84
```
After several tests, the best max_depth is in range [9, 20]. 'max_depth' have a effective influnce on the performance of decision tree model. And it is easy to overfitting in training dataset. Comparing with NBC, it have a better performance in traing dataset.  


# Question 3
In [5.08 Random Forests](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb), VanderPlas performs digits classification with a random forest. He uses n_estimators=1000. Use a validation curve to investigate the choice of n_estimators. Comment briefly on the results (including comparison with results above).
```
make Q3
```


<img src="figs/Validation curve for Random Forest.png" width="500">

```
Best_accuracy: 0.97
Best_parameters: {'n_estimators': 86}
```
```
Train accuracy: 1.00
Test accuracy: 0.98
```
n_estimators=1000 is not necessary for this model.  n_estimators from 50 already can get a good performance. the best n_estimator is around 80. and under best parameters, Random Forest can get a higher accuracy than Decision Tree.

# Question 4

Investigate use of [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). Look at the scikit-learn [adaboost example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html) for ideas. Boosting is discussed in Section 8.2.2 (p345) if ISLR2. Comment briefly on results and your choice of hyperparameters (including comparison with results above).
```
make Q4
```

Validation curve of adaboost in max_depth：



<img src="figs/validation curve of adaboost in max_depth.png" width="500">

Validation curve of adaboost in n_estimators：


<img src="figs/validation curve of adaboost in n_estimators.png" width="500">


Validation curve of adaboost in learning——rate：


<img src="figs/validation curve of adaboost in learning rate.png" width="500">






As for other Hyperparameters, by using gridseachCV to get the best Hyperparameters step by step:  `n_estimators`, `learing_rate`, `algorithm`.

Finally:
```
Best_accuracy: 0.98
Best_parameters: {'algorithm': 'SAMME.R', 'base_estimator__max_depth': 10, 'learning_rate': 1.3000000000000003, 'n_estimators': 150}
```

```
Train accuracy: 1.00
Test accuracy: 0.99
```
Adaboost slightly out performs Random Forest


# Question 5
Adapted the use of SVC in cells 18-26 of Labeled Faces in [the Wild demo in VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb). When selecting optimal hyperparameters, make sure that your range encompasses the minimum. Comment briefly on results and your choice of hyperparameters (including comparison with results above).
```
make Q5
```



```
parameters = {'C': [1,2,3,4,5],
              'gamma': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0008, 0.001, 0.005],
              'kernel': ['linear', 'poly', 'rbf']
              }
```


Validation curve of SVC in C：

<img src="figs/validation curve of SVC in C.png" width="500">


Validation curve of SVC in gamma：

<img src="figs/validation curve of SVC in gamma.png" width="500">


```
Best_accuracy: 0.99
Best_parameters: {'C': 3, 'gamma': 0.001, 'kernel': 'rbf'}
```
```
Train accuracy: 1.00
Test accuracy: 0.99
```

SVC tain faster than adaboost and randomforest. Without any tuning of hyperparameters, SVC already can get a higher accuracy. After tuning the best parameters, SVC can get a best performance. SVC may be the most suitable model for this dataset comparing with models above.  
