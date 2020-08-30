# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:42:56 2020

@author: Arnob
"""
import pandas as pd
import numpy as np
from pandas import set_option
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler, MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


set_option('display.width', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# In[]

raw_data = pd.read_csv("Train_data.csv")

dataframe = raw_data.copy()

# In[]
dataframe = dataframe.drop(dataframe.columns[0], axis =1)


# In[]

dataframe.info()  #Gender have missing data

# In[]
dataframe = dataframe.drop(['gender'], axis = 1) #Dropped Gender

# In[]
desc = dataframe.describe()

# In[]
''' Check point '''
dataframe_1 = dataframe

# In[]
label_encoder = LabelEncoder()

#dataframe["registraton_mode_b"] = label_encoder.fit_transform(dataframe["registrationMode"]) 
# In[]
dataframe = dataframe.drop(['registrationMode'], axis = 1)

# In[]

vals = dataframe.columns.values
print(list(vals))


# In[]
col_names = ['code', 'clientType' ,'planName' ,'accident', 'duration', 'country', 'netSales',
 'netProfit', 'age', 'registraton_mode_b']

# In[]
col_names_reordered = ['code', 'clientType', 'planName'
                       , 'duration', 'country', 'netSales', 
                       'netProfit', 'age']
# In[]
dataframe = dataframe[col_names_reordered] # accident column reordered

# In[]
dataframe_2 = dataframe
# In[]
dataframe = dataframe_2
# In[]

dataframe = pd.get_dummies(dataframe, columns=['code','country'])



# In[]
acc_count = dataframe.groupby('accident').size()
print(acc_count)

 
# In[]
#Label encoding

dataframe['code'] = label_encoder.fit_transform(dataframe["code"])
dataframe['clientType'] = label_encoder.fit_transform(dataframe["clientType"]) 
dataframe['planName'] = label_encoder.fit_transform(dataframe["planName"]) 
dataframe['country'] = label_encoder.fit_transform(dataframe["country"]) 
# In[]
print(dataframe.corr())
'''
drop - regis_mode
'''

# In[]
dataframe = dataframe.drop(['registraton_mode_b'], axis = 1)

# In[]
print(dataframe.skew())

#dataframe.hist()
# In[]
scaled_features = dataframe.copy()
col_names = ['duration']
features = scaled_features[col_names]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)
# In[]
scaled_features[col_names] = features
# In[]
dataframe = scaled_features

# In[] 
# Log Tranform
dataframe['duration'] = np.log(dataframe['duration'])

# In[]
dataframe3= dataframe
# In[]
#PCA application
pca = PCA(n_components=5)
arr = dataframe.values
fit = pca.fit(arr[:,0:9])
# In[] 
arr = dataframe.values
X = arr[:,0:7]
Y = arr[:,7]
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
# In[]
dataframe = dataframe.drop(['clientType'],axis=1)


# In[]
dataframe.to_csv("Test_data_processed_5_5.csv", index = False)


















