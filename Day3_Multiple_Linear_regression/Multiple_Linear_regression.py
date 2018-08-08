#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:07:21 2018

@author: liuchuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[: , 1:]  ### 避免虚拟变量陷阱 

from sklearn.cross_validation import train_test_split   ## 分割数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) ## 多重线性回归模型应用到训练集

y_pred = regressor.predict(X_test)

plt.scatter(np.arange(10),Y_test, color = 'red',label='y_test')
plt.scatter(np.arange(10),y_pred, color = 'blue',label='y_pred')
plt.legend(loc=2);
plt.show()