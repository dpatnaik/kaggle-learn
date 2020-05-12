import data_io
from features import FeatureConverter

def main():
    print("Loading the test data")
    classifier = data_io.load_model()
    
    print ("Load test data. And Clean..")
    test = data_io.get_test_df()
    test = FeatureConverter().clean_data(test)
    passengerIds = test['PassengerId']
    test.drop(['PassengerId'], axis = 1, inplace = True)
    test = test.values
    
    print("Making predictions") 
    predictions = classifier.predict(test).astype(int)
    #predictions = predictions.reshape(len(predictions), 1)
    
    print("Writing predictions to file")
    data_io.write_submission(predictions, passengerIds, ['PassengerId', 'Survived'])

if __name__=="__main__":
    main()