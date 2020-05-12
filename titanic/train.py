import data_io
from features import FeatureConverter
from classifiers import EnsembleClassifier

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def main():
    
    classifier1 = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=5.0)
    classifier2 = DecisionTreeClassifier(max_depth = 10, criterion = 'entropy', random_state = 0)
    classifier3 = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
    classifier4 = SVC(kernel = 'rbf', C = 10.0, random_state = 0, gamma = 0.10)
    classifier5 = LogisticRegression(penalty = 'l2', C = 1.0, random_state = 0)
    classifier6 = GaussianNB()
        
    print("Reading in the training data")
    train = data_io.get_train_df()
    
    print ("Cleaning data. Check here for imputation, One hot encoding and factorization procedures..")
    train = FeatureConverter().clean_data(train)
    train.drop(['PassengerId'], axis = 1, inplace = True)
    #print train.head()
    train = train.values
    
    eclf = EnsembleClassifier(clfs = [classifier1, classifier2, classifier3, classifier5, classifier6], voting = 'hard')
    #eclf = EnsembleClassifier(clfs = [classifier2], voting = 'hard')    
    scores = cross_val_score(estimator = eclf, X = train[0:,1:], y = train[0:,0], cv = 10, scoring = 'roc_auc')
    
    print("Accuracy: %0.4f (+/- %0.3f)" % (scores.mean(), scores.std()))
    eclf.fit(train[0:,1:],train[0:,0])

    print("Saving the classifier")
    data_io.save_model(eclf)
    
if __name__=="__main__":
    main()