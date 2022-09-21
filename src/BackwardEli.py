# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:14:37 2022

@author: shiva
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

data_set=pd.read_csv('MLRdata.csv')

x=data_set.iloc[:,:-1]
y=data_set.iloc[:,4:]


# Important and to be used in place of LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x=ct.fit_transform(x)

#countering the dummy variable trap
x=x[:,1:]

#splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

#Training the Model
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train) 

#predicting the data
x_pred=regressor.predict(x_train)
y_pred=regressor.predict(x_test)

#testing the score
print('train score: ',regressor.score(x_train,y_train))
print('test score: ',regressor.score(x_test,y_test))

#process for backward elimination
import statsmodels.api as smf
x=np.append(arr=np.ones((21,1)).astype(int),values=x,axis=1)

#x_opt is array of features affecting the dependent features most
x_opt=x[:,[0,1,2,3,4,5]]

regressor_OLS=smf.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())





