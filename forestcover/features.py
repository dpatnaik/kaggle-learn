import re
import sys
import sklearn.preprocessing as preprocessing
import pandas as pd

class FeatureConverter:
    
    def clean_data(self, df):
        
        # StandardScaler will subtract the mean from each value then scale to the unit variance
        scaler = preprocessing.StandardScaler()
        
#         df['Elevation'] = scaler.fit_transform(df['Elevation'].astype(float))
#         df['Horizontal_Distance_To_Hydrology'] = scaler.fit_transform(df['Horizontal_Distance_To_Hydrology'].astype(float))
#         df['Vertical_Distance_To_Hydrology'] = scaler.fit_transform(df['Vertical_Distance_To_Hydrology'].astype(float))
#         df['Horizontal_Distance_To_Roadways'] = scaler.fit_transform(df['Horizontal_Distance_To_Roadways'].astype(float))
#         df['Horizontal_Distance_To_Fire_Points'] = scaler.fit_transform(df['Horizontal_Distance_To_Fire_Points'].astype(float))
#         df['Aspect'] = df['Aspect'] / 360.0
#         df['Slope'] = df['Slope'] / 90.0
#         df['Hillshade_9am'] = df['Hillshade_9am'] / 255.0
#         df['Hillshade_Noon'] = df['Hillshade_Noon'] / 255.0
#         df['Hillshade_3pm'] = df['Hillshade_3pm'] / 255.0
        
        #print df['Hillshade_9am'].describe()
        #print df['Hillshade_Noon'].describe()
        #print df['Hillshade_3pm'].describe()
        #print df['Slope'].describe()
        
        #print df['Elevation'].describe()
        
        #print df.columns
        #sys.exit()
        return df