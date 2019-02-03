# SVM 学习

![](https://ws3.sinaimg.cn/large/006tNc79ly1fzsix8oem7j30m81jk7wh.jpg)

----

## 1.SVM 基础了解


### What is SVM ？

* 支持向量机是一种监督的机器学习，可用于分类和回归问题，主要用于分类问题

* 在 SVM 中，我们把每个数据项看作 高维空间的一个坐标点（空间维度对应与数据的特征数）

* 支持向量机的学习策略就是 间隔最大化，形式化为求解一个凸二次规划，学习算法就是凸二次规划的最优化算法


### How is the data classified

* 通过寻找一个能够分开两类数据的超平面 从而实现分类

* 要寻找一个最优化的 超平面

* 超平面到每一类的数据的距离都是最大的 即为最优化


### SVM 模型

* 线性可分支持向量机 ： 训练数据线性可分， 通过硬间隔最大化(harf margin maximization) 学习一个线性分类器

* 线性支持向量机 ：训练数据线性**近似**可分， 通过软间隔最大化(soft margin maximization) 学习一个线性分类器

* 非线性支持向量机 ： 训练数据线性不可分，使用核技巧(kernel kick) 进行分类


------


## 2.SVM 详述

### 1. 线性可分支持向量机

> 数据集的线性可分性 （2分数据）：存在某个超平面能将两类实例点完全正确（所有实例点）的划分到超平面的两侧，称数据集线性可分，否则线性不可分



* 支持向量机的学习都是在特征空间完成的，将输入从输入空间**线性或者非线性**映射到特征空间，生成特征向量。

* 支持向量机通过间隔最大化求解最优的超平面，所以最终解是唯一的



#### 1. 间隔

> 一个点距离超平面的远近可以表示分类预测的确信程度

![](https://ws4.sinaimg.cn/large/006tNc79ly1fztb0ri1gyj30jy0h0jt3.jpg)

1. **函数间隔** ： 在数据集和超平面确定的情况下，使用$y*(w*x+ b)$ 表示分类的正确性和可信度，所以$y*(w*x+ b)$ 就是函数间隔

2. 选择分离超平面的时候不能只看函数间隔，因为等比例的改变 w,b 超平面没有变函数间隔却改变了，需要规范化参数，使得间隔是确定的，即为**几何间隔**。

![](https://ws1.sinaimg.cn/large/006tNc79ly1fztb9lfhxzj30ms05074t.jpg)

3. **间隔最大化** ： 对训练数据集寻找几何间隔最大化的的超平面，即以充分大的确信度对数据进行分类。这样的超平面对未知的数据有较好的预测能力

#### 2. 支持向量机的学习算法 --- 最大间隔法

* 问题转化为一个约束最优化问题 ： 最大化间隔 r ，满足所有的点都间隔都大于 r

![](https://ws2.sinaimg.cn/large/006tNc79ly1fztbylm3wrj30rs068756.jpg)

* 函数间隔 r 的取值不影响最优化问题的解，取 r = 1； 同时最大化 $\frac{1}{||w||}$ 等价于 最小化 $1/2 ||w||^{2}$

![](https://ws1.sinaimg.cn/large/006tNc79ly1fztbyzjaljj30rs068gmi.jpg)


* 通过上述的约束优化，求解得到超平面 $$ w*x + b = 0 $$

* 分类决策函数 $$ f(x) = sign(w*x + b)$$



#### 3. 支持向量

> 线性可分的样本中，样本点距离超平面最近的样本点的实例称为 **支持向量**
> H1 和 H2 上面的点就是支持向量
![](https://ws3.sinaimg.cn/large/006tNc79ly1fztcahpqz6j30rs0fy76o.jpg)


* H1 和 H2 之间的距离称为 间隔(margin)
* H1 和 H2 称为 间隔边界

**决定分离超平面的时候，只有支持向量起作用，其他的实例点并不起作用；所以分类模型称为支持向量机**



#### 4. 线性可分支持向量的学习算法 --- 对偶算法
> 类似于热力学统计物理中的拉格朗日算子 方法(名字可能记得不是很准确)

----

### 2. 线性支持向量机 --- 软间隔最大化

> 训练数据往往都是线性不可分的，如噪声，异常点等等
> 数据线性不可分，那么约束的不等式就不成立了， 就需要修改为软间隔最大分类


#### 1. 松弛变量

* 线性不可分意味着，对于某些点，不满足约束条件 $$y*(w*x + b) > 1$$

* 对于每一个样本点，引入一个松弛变量 l >= 0 ; $$y*(w*x + b) + l > 1$$

* 对于每一个松弛变量，在目标函数中对应一个代价，所以目标函数变为：

![](https://ws3.sinaimg.cn/large/006tNc79ly1fztd6zni4gj30rs04474o.jpg)

* C> 0  称为惩罚参数， C 值大，对于误分类的惩罚大。

* 最小化目标函数： 间隔尽量大 + 误分类的点尽量少



#### 2. 支持向量

![](https://ws2.sinaimg.cn/large/006tNc79ly1fztfr5f71aj30rs0k2wic.jpg)

#### 3. 合叶损失函数

* SVM 算法的另一种解释 --- 暂不详述


-----

### 3. 非线性支持向量机和 核函数


#### 1. 核技巧(kernel trick)

> 通过非线性变换，将非线性问题变换为线性问题

> 核技巧应用到 SVM 中的核心思想 ： 通过一个非线性变换将输入空间转 对应到一个特征空间，使得输入空间的超曲面模型转换为 特征空间的超平面模型-- SVM


#### 2， 核函数定义

* X 为输入空间， H 为特征空间存在一个 X 到 H 的映射； $\phi (x)  X--> H$

* 对于 $ x,z \in X  , K(x,z) = \phi (x) * \phi(z)$

* K(x,z)  就是核函数， 使用的时候，不显示的定义 映射函数，而是直接 **依据领域知识选取核函数**


#### 3. 正定核函数 (positive definite kernel function)

> 暂不详述


#### 4. 常用核函数

* 多项式核函数 ： p次多项式分类器

* 高斯核函数 ： 高斯径向基函数分类器、

* 字符串核函数


----

### 4. SMO 优化算法 -- 序列最小最优化算法


> 暂不详述


----

## 3. 算法实现

### 1. 实验数据

![](https://ws4.sinaimg.cn/large/006tNc79ly1fzti6zqiutj30rc0t6tcc.jpg)

* 使用 Age 和 salary 的数据 去预测是否购买 --- 二维数据便于绘制图形

#### Importing the libraries

```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

#### Importing the dataset
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```

#### Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split 将不再使用
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```


#### Feature Scaling --- 特征规范， 不同特征的 大小不同，规范化
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


#### Fitting SVM to the Training set

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0,C = 2)
classifier.fit(X_train, y_train)
```


#### Predicting the Test set results
```python

y_pred = classifier.predict(X_test)
```

####  Visualising the Training set results
```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

#### Visualising the Test set results

```python

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## 结果

![](https://ws3.sinaimg.cn/large/006tNc79ly1fztio39g7ej30as07qdg0.jpg)

![](https://ws2.sinaimg.cn/large/006tNc79ly1fztio8vyxaj30as07qmx6.jpg)

## 调参数

*  测试集分割部分，参数的选取

*  SVC kernel中参数调节



# 附录

##### [SVM 实现的 jupyter notebook](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day11-12-13-14_SVM/SVM%20%E5%AE%9E%E7%8E%B0.ipynb)

##### [SVM 实现 python 代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day11-12-13-14_SVM/SVM_sklearn.py)

##### [数据集](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/Social_Network_Ads.csv)


# 参考

1. 统计学习方法 --- 李航

2. https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md

3. 机器学习实战
















