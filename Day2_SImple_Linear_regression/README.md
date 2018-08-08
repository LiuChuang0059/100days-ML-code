# 简单线性回归

![](https://github.com/LiuChuang0059/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%202.jpg)


## 1. goal

### 1 使用单一特征 预测 结果（response）
* 假定两个变量线性相关，尝试寻找一个线性方程 来进行尽可能精确的预测

### 2寻找最好的拟合直线
* 最小化 实际观测值$y_{i}$和预测值$y_{p}$之间的长度
$$min{sum(y_{i}-y_{p})^{2}}$$

### 3 回归任务
预测学生的 分数（百分制）  与他们学习时间之间的关系
$$score = b_{0} + b_{1}* hours$$



## 2. 实操

![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day2_SImple_Linear_regression/%E6%95%B0%E6%8D%AE%E9%9B%86.png)
### Step 1: 预处理数据-----类似于[Day1](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day1_Data_preprocessing/README.md)
1. 导入库
2. 导入数据集
3. 检查缺失数据
4. 分割数据集
5. 特征缩放-----使用专有的库

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 


```

### Step 2: Fitting Simple Linear Regression Model to the training set
* 把数据集 拟合到简单线性回归模型  使用fit函数
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() ## 创建一个regressor 对象
regressor = regressor.fit(X_train, Y_train) ### 对象拟合到数据集里面
```

### Step 3: Predecting the Result
* 在训练好的regressor使用预测模型
* 结果输出到向量Y_pred

```python
Y_pred = regressor.predict(X_test)
```


### Step 4: Visualization
```python
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')   ## 训练结果
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')   ## 测试结果
```
**ps: 结果可视化中 测试集的结果使用Y_test因为y_pred上的点和预测曲线高度重合**
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day2_SImple_Linear_regression/%E6%B5%8B%E8%AF%95%E9%9B%86%E7%BB%93%E6%9E%9C.png)

## 3.结果
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day2_SImple_Linear_regression/Day%202_result.png)
### [1 所需数据集](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day2_SImple_Linear_regression/studentscores.csv)
### [2 完整代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day2_SImple_Linear_regression/Simple_linear_regression.py)


# 小问题

## 1 数学公式显示
安装 chrome中的 [Google with mathjax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related) 插件
参考：https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related
