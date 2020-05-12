import sys
import numpy as np
import data_io
from features import FeatureConverter

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC

rf_params_grid = \
{
    'max_features': [0.5, 1.],
    'max_depth': [5., None]
}

lr_params_grid = \
{
    'C': [0.001, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2']
}

svc_params_grid = \
{
    'C': [0.001, 0.1, 1.0, 10.0, 100.0],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.1, 1.0, 10.0, 100.0]
}

if __name__=="__main__":
    
    print("Reading in the training data")
    train = data_io.get_train_df()
    
    print ("Cleaning data. Check here for imputation, One hot encoding and factorization procedures..")
    train = FeatureConverter().clean_data(train)
    train.drop(['PassengerId'], axis = 1, inplace = True)
    #print train.head()
    train = train.values
    
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 100), rf_params_grid, cv = 5, verbose = 1)
    grid_search.fit(train[0:,1:],train[0:,0])
    
    print grid_search.best_params_
    
    grid_search = GridSearchCV(LogisticRegression(random_state = 0), lr_params_grid, cv = 5, verbose = 1)
    grid_search.fit(train[0:,1:],train[0:,0])
    
    print grid_search.best_params_
    
    grid_search = GridSearchCV(SVC(random_state = 0), svc_params_grid, cv = 5, verbose = 1)
    grid_search.fit(train[0:,1:],train[0:,0])
    
    print grid_search.best_params_