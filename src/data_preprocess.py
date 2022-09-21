# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
imputer=SimpleImputer(strategy='mean')
data=pd.read_csv('dataset.csv')
lencox=LabelEncoder()

x=data.iloc[:,:-1].values
y=data.iloc[:,3].values
#imputer=imputer.fit(x[:, 2:3]) 
x[:,2:3]=imputer.fit_transform(x[:,2:3])
#x[:, 0]= lencox.fit_transform(x[:, 0])

#using CT in place of OneHotEncoder and LabelEncoder
ct=ColumnTransformer([("country", OneHotEncoder(), [0])], remainder = 'passthrough')
x=ct.fit_transform(x)
#onenco= OneHotEncoder(categorical_features=[0])
#x=onenco.fit_transform(x).toarray()
print(x) 
#print(y)