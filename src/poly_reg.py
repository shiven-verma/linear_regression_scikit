# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:36:02 2022

@author: shiva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_set=pd.read_csv('salary_dat.csv')

x=data_set.iloc[:,0:1]
y=data_set.iloc[:,1]

#For linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#features from preprocessing module for Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
#degree gives the power of elements of x
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color='b')
plt.plot(x,lin_reg_2.predict(x_poly),color='r')
plt.show()

lin_pred=lin_reg.predict([[6.5]])
poly_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(lin_pred)
print(poly_pred)
