import re
import sys
import sklearn.preprocessing as preprocessing
import pandas as pd

class FeatureConverter:
    
    def clean_data(self, df):
            
        # Fill Age NaNs with 0.0
        df['Age'].fillna(0, inplace=True)
    
        # Replace 0 age with average age in that gender and Pclass combination
        grouped = df['Age'].groupby([df['Pclass'], df['Sex']])
        medians = grouped.median().to_dict()
        df.ix[df.Age == 0, 'Age'] = df.apply(lambda row: round(medians[row['Pclass'], row['Sex']], 0), axis = 1)
    
        #df.ix[df.Age <= 13, 'Sex'] = df.apply(lambda row: 'male-child' if row['Sex'] == 'male' else row['Sex'], axis = 1)
        #df.ix[df.Age >= 45, 'Sex'] = df.apply(lambda row: 'male-old' if row['Sex'] == 'male' else row['Sex'], axis = 1)
 
        # StandardScaler will subtract the mean from each value then scale to the unit variance
        scaler = preprocessing.StandardScaler()
        #df['Age_scaled'] = scaler.fit_transform(df['Age'])
        
        # Divide all fares into quartiles
        df['Age_bin'] = pd.qcut(df['Age'], 6)
        #df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)
                
        # Create a dataframe of dummy variables for each distinct value of 'Embarked'
        df = pd.concat([df, pd.get_dummies(df['Sex']).rename(columns=lambda x: str(x))], axis=1)
    
        # Check whether 'Embarked' has no NaN. If any has, analyze based on ticket number and cabin number
        # Mr. Ostby embarked from 'C'
        df.ix[df['Embarked'].isnull(), 'Embarked'] = 'C'
        # Dummy Variables (One hot encoding)
        #df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
        
        # Replace NaN fares with median fare in that gender and Pclass combination
        grouped_fare = df['Fare'].dropna().groupby([df['Pclass'], df['Sex']])
        mean_fare = grouped_fare.mean().to_dict()
    
        df.ix[df.Fare.isnull(), 'Fare'] = df.apply(lambda row: round(mean_fare[row['Pclass'], row['Sex']], 2), axis = 1)
        df['Fare'] = df['Fare'].round(1)
            
        df['Ticket_Number'] = df['Ticket'].map(lambda x : re.compile("\d{3,7}").search(x).group() if re.compile("\d{3,7}").search(x) else 0)
        #df = df.sort_values(by ='Ticket', axis = 0, ascending = True)
        
        grouped_fare = df['Fare'].groupby(df['Ticket_Number'])
        mean_fare = (grouped_fare.mean()/ grouped_fare.size())
        
        df.ix[df.Fare.notnull(), 'Fare'] = df.apply(lambda row: round(mean_fare[row['Ticket_Number']], 2), axis = 1)
        df['Fare'] = df['Fare'].round(1)
        
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
                
        # Divide all fares into quartiles
        df['Fare_bin'] = pd.qcut(df['Fare'], 4)
        #df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
            
        df = pd.concat([df, pd.get_dummies(df['Pclass']).rename(columns=lambda x: 'Class_' + str(x))], axis=1)
 
        df['Title'] = df['Name'].map(lambda x : re.compile(r'(\w+\.)').findall(x)[0][:-1])
        # Group low-occuring, related titles together
        df.ix[df.Title == 'Jonkheer', 'Title'] = 'Master'
        df.ix[df.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'
        df.ix[df.Title == 'Mme', 'Title'] = 'Mrs'
        df.ix[df.Title.isin(['Capt', 'Rev', 'Dr', 'Don', 'Major', 'Col', 'Sir']), 'Title'] = 'Sir'
        df.ix[df.Title.isin(['Dona', 'Lady', 'Countess']), 'Title'] = 'Lady'
        # Build binary features
        #df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
        
        #df['female_Parch'] = df['female'] *  df.Parch
        #df['female_SibSp'] = df['female'] *  df.SibSp
        
        #df['male_Parch'] = df['male'] *  df.Parch
        #df['male_SibSp'] = df['male'] *  df.SibSp
        
        #df['male-old_Parch'] = df['male-old'] *  df.Parch
        #df['male-old_SibSp'] = df['male-old'] *  df.SibSp
        
        #df['male-child_Parch'] = df['male-child'] *  df.Parch
        #df['male-child_SibSp'] = df['male-child'] *  df.SibSp
    
        # Replace missing values with "U0"
        df.ix[df.Cabin.isnull(), 'Cabin'] = 'U0'        
        # create feature for the alphabetical part of the cabin number
        df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
        # convert the distinct cabin letters with incremental integer values
        df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
        
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        df['Individual'] = 0
        df.ix[df['FamilySize'] == 1, 'Individual'] = 1
        
        df['Small_Family'] = 0
        df.ix[df['FamilySize'] <= 4, 'Small_Family'] = 1
        df.ix[df['FamilySize'] == 1, 'Small_Family'] = 0
        
        df['Big_Family'] = 0
        df.ix[df['FamilySize'] > 4, 'Big_Family'] = 1
        
        #print df['Individual'].groupby([df['Survived'], df['Sex']]).value_counts()
        
        # Drop Embarked, Cabin and Ticket columns
        df.drop(['Embarked', 'Age', 'Title', 'Name', 'Sex', 'Fare', 'Cabin', 'Ticket', 'Pclass'], axis = 1, inplace = True)
        df.drop(['Age_bin', 'Fare_bin', 'FamilySize', 'Ticket_Number'], axis = 1, inplace = True)
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
        #df.drop(['Individual', 'Small_Family', 'Big_Family'], axis=1, inplace=True)
        df.drop(['CabinLetter'], axis=1, inplace=True)
        
        print df.columns
        return df