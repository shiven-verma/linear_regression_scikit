# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:12:44 2022

@author: shiva
"""
#run in Jupyter notebook, Sympy is not working in spyder
import numpy as np
import pandas as pd
from sympy import *
data=pd.read_csv("radata.csv")
import matplotlib.pyplot as plt

def grad(x,y,k):
    m=len(y)
    costf=(1/2*m)*np.sum((y-k*x)**2)
    return costf
    
loop=0
x=data.iloc[:,:-1]
y=data.iloc[:,1]

thetaval=0.5
k=symbols('k')
diff=Derivative(grad(x, y, k),k)
#print(grad(x,y,k))
#print(diff.doit())
#dcon=diff.doit().replace(k,thetaval)
#print(dcon)
loop=0
while loop<40:
    thetaval=thetaval-0.00005*diff.doit().replace(k,thetaval)
    loop=loop+1
    
print(thetaval)
#print('dcon : ',dcon)

plt.scatter(x,y)
plt.plot(x,x*thetaval,'r')
plt.show()