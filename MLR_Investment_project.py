import numpy as np 
import pandas as  pd 
import matplotlib.pyplot as plt 

#loading dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\NIT All-Projects\Investment_ML_project\Investment.csv")

#dependent variable
x = df.iloc[:,:-1]
#independent variable
y = df.iloc[:,4]

x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

#creating MLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

x = np.append(arr = np.ones((50,1)).astype(int), values = x,axis = 1)

import statsmodels.api as sm 
x_opt = x[:,[0,1,2,3,4,5]]
#Ordinary_least_sqares
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm 
x_opt = x[:,[0,1,2,3,5]]
#Ordinary_least_sqares
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()



