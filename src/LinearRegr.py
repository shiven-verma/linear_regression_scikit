# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:03:46 2022

@author: shiva
"""
#data libraries in python
import numpy as np
import pandas as pd
#plotting Library in Python
import matplotlib.pyplot as mtp
#SkLearn ML library 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#importing the data
data_set=pd.read_csv("radata.csv")

#splitting the data in dependent and independent variable
x=data_set.iloc[:,:-1]
y=data_set.iloc[:,1]

#splitting the data in training and test set
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)

# Standard Scaling of data
#from sklearn.preprocessing import StandardScaler
#st_x = StandardScaler()
#x_train = st_x.fit_transform(x_train)
#x_test =st_x.transform(x_test)

#training the model 
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the dependent variable x for training dataset, y for test dataset
x_pred=regressor.predict(x_train)
y_pred=regressor.predict(x_test)

#plotting the data
mtp.scatter(x_train, y_train, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()   