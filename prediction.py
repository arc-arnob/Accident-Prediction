# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:50:17 2020

@author: Arnob
"""
# In[]
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
filename = 'Train_data_processed_5_5.csv'
test_file_name = 'Test_data_processed_5_5.csv'

# In[]
dataframe = read_csv(filename)
array = dataframe.values
X = array[:,0:7]
Y = array[:,7]
# In[]

test_size = 0.50

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
)
# In[]
criterion = ['gini', 'entropy']
max_depth = [4,6]
estimators = [30,40,50]
rates = [0.1,0.01,1]
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)

# In[]
parameters = dict(base_estimator__criterion=criterion,
                      base_estimator__max_depth=max_depth,
                      n_estimators = estimators,
                      learning_rate = rates)
 
model = AdaBoostClassifier(base_estimator=dtc, learning_rate= 0.01, n_estimators= 30, random_state= None)
# In[]
grid = GridSearchCV(estimator=model, param_grid=parameters)


grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.get_params())
#n_components = 1, priors =None, shrinkage = None, solver = 'svd', store_covariance= False, tol= 0.0001
# In[]
#LDA
lda = LinearDiscriminantAnalysis(n_components = 1, priors =None, shrinkage = None, solver = 'svd', store_covariance= False, tol= 0.0001)

# In[]
lda_param_grid = {"solver" : ["svd","lsqr","eigen"],
              "tol" : [0.0001,0.0002,0.0003],
              "n_components": [1,2,3,4,5,6,7,8]}
                
grid = GridSearchCV(estimator=lda, param_grid=lda_param_grid)


grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.get_params())





# In[]
mod = LogisticRegression()
mod.fit(X_train, Y_train)
result = mod.score(X_test, Y_test)







# In[]

lda.fit(X_train, Y_train)
result = lda.score(X_test, Y_test)

# In[]
scoring = 'roc_auc'
models = []
results = []
names=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('ADA',AdaBoostClassifier(DecisionTreeClassifier())))

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)


    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# In[]
print(result*100.0)

# In[]
kfold = KFold(n_splits=10, random_state=7)
scoring = 'roc_auc'
# In[]
results = cross_val_score(mod,X,Y,cv=kfold,scoring=scoring)
print(results.mean(), results.std())

# In[]
df_test = read_csv(test_file_name)
grid = df_test.values
# In[]
y_pred = mod.predict_proba(grid)[:,1]


# In[]
numpy.savetxt("res_yash.txt",y_pred, fmt='%-7.2f')