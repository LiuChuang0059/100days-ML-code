#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:45:09 2018

@author: liuchuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() ## 创建一个regressor 对象
regressor = regressor.fit(X_train, Y_train) ### 对象拟合到数据集里面

Y_pred = regressor.predict(X_test)

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')   ## 训练结果
plt.scatter(X_test , Y_test, color = 'green')
plt.plot(X_test , regressor.predict(X_test), color ='blue')   ## 测试结果