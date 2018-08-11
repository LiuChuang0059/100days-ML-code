# 逻辑回归

1[](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/Day%204.jpg)
# 1.逻辑回归 简要介绍

## 1 应用
* 逻辑回归 用来处理不同的分类问题，----预测当前的值 属于哪个组
* 给出一个类似于二进制结果（0，1）的输出 -----离散的输出结果（不同于线性回归的连续结果）

## 2 如何工作
* 使用基础逻辑函数 通过估算概率来计算输出结果和一个或者多个特征之间的关系

## 3 作出预测
* 将概率值转换为二进制的数 以便进行预测
* 使用sigmoid函数 获得（0，1）的范围内结果
* 阈值分类器 将（0，1）范围内的结果 转化为0或者1

## 4 sigmoid函数
$$\theta(s)=\frac1{1+e^{-s}}$$

## 5 极大似然估计
利用已知的样本结果，反推最有可能（最大概率）导致这样结果的参数值。

--------------
---------------

# 2. 基本原理
## 1 构造一个预测函数（h函数） ：猜测预测函数的“大概”形式，比如是线性函数还是非线性函数。

### 1 使用 sigmoid函数
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/sigmoid.png)

### 2 确定边界类型
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/decision_boundry.jpeg)
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/Nonlinear_decision_boundray.jpeg)


### 3线性边界

1. 边界线

$$\Theta ^{T}=\Theta_{0}+ \Theta_{1}x_{1}+\Theta_{2}x_{2}+...+\Theta_{n}x_{n}$$

2.构造h函数
$$h_{\Theta }(x)=\frac{1}{1+e^{-\Theta ^{T}x}}$$
h函数表示结果取1 的概率
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/h(x).png)

## 2 构造cost 函数：表示预测的输出（h）与训练数据类别（y）之间的偏差
> 综合考虑所有训练数据的“损失”，将Cost求和或者求平均，记为J(θ)函数，表示所有训练数据预测值与实际类别的偏差。
![]()

### 1 使用最大似然估计推导 ---[详细](https://blog.csdn.net/ligang_csdn/article/details/53838743)

### 2 结果
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/cost%E5%87%BD%E6%95%B0.jpeg)

## 3 使用梯度下降法求 J（theta）最小值 or 梯度上升求最大值

### 1梯度下降法表示theta更新过程
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.jpeg)

结果：
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%BB%93%E6%9E%9C.jpeg)



## 4 梯度下降过程向量化
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%90%91%E9%87%8F%E5%8C%96.png)

----------------
-----------------


# 3 实操
## 1.实例
![](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/data.PNG)
> 尝试预测哪些用户会购买这种全新SUV。并且在最后一列用来表示用户是否购买。我们将建立一种模型来预测用户是否购买这种SUV，该模型基于两个变量，分别是年龄和预计薪资。因此我们的特征矩阵将是这两列。我们尝试寻找用户年龄与预估薪资之间的某种相关性，以及他是否购买SUV的决定。

## Step 1  Data Pre-Processing-----[详见](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day1_Data_preprocessing/README.md)

```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values     ## 选取2，3两列--Age+ salary
y = dataset.iloc[:, 4].values      ## 选取最后一列

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler    ##特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```

> 
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/%E6%95%B0%E6%8D%AE%E9%9B%86%E6%95%A3%E7%82%B9%E5%9B%BE.png)

## Step 2  Logistic Regression Model
* 逻辑回归应用于分类好的数据集
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```
> LogisticRegression()函数有很多参数，详见[link](https://blog.csdn.net/CherDW/article/details/54891073)
## Step 3 Predection

```python
y_pred = classifier.predict(X_test)

```

## Step 4  Evaluating The Predection-------混淆矩阵等待解决？？？？

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

## Step 5 visulisation
> 原文无代码,但是有可视化结果，自行可视化-----其中[plot_decision_regions函数代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/plot_decison_regions.py)
```python
plot_decision_regions(X_test, y_pred, classifier=cl)
plt.xlabel('age')
plt.ylabel('salary')
plt.legend(loc='upper left')
plt.title("Test set")
plt.show()

```

## 结果
![training-set](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/training_set.png)
![test-set](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/Test-set.png)

----------------------
---------------------


# 4

## 1 [完整代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/logistic_regression.py)
## 2 [数据集](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/Social_Network_Ads.csv)


# 参考
1. 极大似然估计 ： https://blog.csdn.net/zengxiantao1994/article/details/72787849
2. 数学推导以及图片来源： https://blog.csdn.net/ligang_csdn/article/details/53838743
3. 可视化： https://blog.csdn.net/xlinsist/article/details/51289825
4.logisticRegression() 函数参数： https://blog.csdn.net/CherDW/article/details/54891073
