# SVM 学习

![](https://ws3.sinaimg.cn/large/006tNc79ly1fzsix8oem7j30m81jk7wh.jpg)

----

**文章公式图片摘自 组会报告的 PPT**

#  1.SVM 基础了解


### What is SVM ？

* 支持向量机是一种监督的机器学习，可用于分类和回归问题，主要用于分类问题

* 在 SVM 中，我们把每个数据项看作 高维空间的一个坐标点（空间维度对应与数据的特征数）

* 支持向量机的学习策略就是 间隔最大化，形式化为求解一个凸二次规划，学习算法就是凸二次规划的最优化算法

----

### SVM 模型

- 线性可分支持向量机 ： 训练数据线性可分， 通过硬间隔最大化(harf margin maximization) 学习一个线性分类器
- 线性支持向量机 ：训练数据线性**近似**可分， 通过软间隔最大化(soft margin maximization) 学习一个线性分类器
- 非线性支持向量机 ： 训练数据线性不可分，使用核技巧(kernel kick) 进行分类

------

### How is the data classified

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2be9wookoj30be0auq8z.jpg)

如图所示， 有两类数据，⭕️ 和 三角，由于某些原因，⭕️**表示的数据标签** **y = +1,** 三角表示数据标签 **y= -1**

* 通过寻找一个能够分开两类数据的超平面 从而实现分类

* 要寻找一个最优化的 超平面

* 超平面到每一类的数据的距离都是最大的 即为最优化

超平面函数：

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2becfgdrdj30fg038glr.jpg)

数据点到超平面的距离可以简单表示为：

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bedkg539j30ak03w0su.jpg)

其中 Xp是 X 在超平面上面的正交投影

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2beg2ksjoj30gg05ydgb.jpg)

通过上面式子计算得到 r，为了使得 距离非负数，我们乘上标签值

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2behmmx63j30b205yq38.jpg)

---------------

# 2.SVM 详述

### 1. 线性可分支持向量机

> 数据集的线性可分性 （2分数据）：存在某个超平面能将两类实例点完全正确（所有实例点）的划分到超平面的两侧，称数据集线性可分，否则线性不可分

* 支持向量机的学习都是在特征空间完成的，将输入从输入空间**线性或者非线性**映射到特征空间，生成特征向量。

* 支持向量机通过间隔最大化求解最优的超平面，所以最终解是唯一的

#### 1. 间隔（Margin）

> 一个点距离超平面的远近可以表示分类预测的确信程度

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2beic5ydyj30da09k3zr.jpg)



1. **函数间隔** ： 在数据集和超平面确定的情况下，使用$y*(w*x+ b)$ 表示分类的正确性和可信度，所以$y*(w*x+ b)$ 就是函数间隔

2. 选择分离超平面的时候不能只看函数间隔，因为等比例的改变 w,b 超平面没有变函数间隔却改变了，需要规范化参数，使得间隔是确定的，即为**几何间隔**。

![](https://ws1.sinaimg.cn/large/006tNc79ly1fztb9lfhxzj30ms05074t.jpg)

3. 观察上面的公式，**分子分母上下等比例的变化**，所以我们固定上面的式子

   ![](https://ws4.sinaimg.cn/large/006tNc79ly1g2beqc0pz6j30b2028glo.jpg)

   这个称为 函数间隔，一般便于计算，取为 1

   

   

   　

#### 2. 支持向量机的学习算法 --- 最大间隔法

**SVM的模型是让所有点到超平面的距离大于一定的距离，也就是所有的分类点要在各自类别的支持向量两边，同时距离尽可能的远** ，即以充分大的确信度对数据进行分类。这样的超平面对**未知的数据有较好的预测能力**

* 问题转化为一个约束最优化问题 ： 最大化间隔 r ，满足所有的点都间隔都大于 r

![](https://ws2.sinaimg.cn/large/006tNc79ly1fztbylm3wrj30rs068756.jpg)

* 如上面的分析，函数间隔 r 的取值不影响最优化问题的解，取 r = 1； 同时最大化 $\frac{1}{||w||}$ 等价于 最小化 $1/2 ||w||^{2}$

![](https://ws1.sinaimg.cn/large/006tNc79ly1fztbyzjaljj30rs068gmi.jpg)

**目标函数** **min** **是** **凸函数，** **同时约束条件不等式是仿射函数的，根据凸优化理论，我们可以通过拉格朗日函数将我们的优化目标转化为无约束的优化函数**

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2bevqowyrj30r80acwfy.jpg)

根据 下面两个式子，消掉 L(w,b,a)中的 w和 b，转化为

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bewzhw5qj30lo048t99.jpg)



根据**SMO** 算法，求解出 a向量，再 带回 关系式 求解 w，b

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2bey8j43ej30aa04g3yp.jpg)

对于所有的支持向量 xs，ys ，根据下面的两个公式求解 b，之后取平均值

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bezh75c5j30f403oq35.jpg)

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2bezplcelj30f8040mxe.jpg)

**PS： 观察 W，b 公式，我们不难发现，参数都是由支持向量决定的，这个可以很直观的理解，外围的对于超平面的划分不起作用。**

**决定分离超平面的时候，只有支持向量起作用，其他的实例点并不起作用；所以分类模型称为支持向量机**

#### 3. 支持向量

> 线性可分的样本中，样本点距离超平面最近的样本点的实例称为 **支持向量**
> H1 和 H2 上面的点就是支持向量

![](https://ws3.sinaimg.cn/large/006tNc79ly1fztcahpqz6j30rs0fy76o.jpg)

根据 KKT  互补条件

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2bf1cf0jej30g60440sz.jpg)

所以， **a>0** ,括号内的函数就要等于零，上面的点就都是 支持向量。

#### 4. 新数据预测

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2bf5wpcsej30g6044aac.jpg)

**对于新点** x的预测，只需要计算它与训练数据点的内积即可，由于**非支持向量的点的** **a** 是 **0** **，** 新的内即运算只是计算和支持向量之间



----

### 2. 线性支持向量机 --- 软间隔最大化

1. 训练数据往往都是线性不可分的，如噪声，异常点等等
   数据线性不可分，那么约束的不等式就不成立了， 就需要修改为软间隔最大分类，如下图所示

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bf96f2yjj30ai08kgly.jpg)

2. 某些即使线性可分，但是分割影响模型的泛化

   ![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfa5tmr0j30a309wjs1.jpg)

   


#### 1. 松弛变量

* 线性不可分意味着，对于某些点，不满足约束条件 $$y*(w*x + b) > 1$$

* 对于每一个样本点，引入一个松弛变量 l >= 0 ; $$y*(w*x + b) + l > 1$$

* 对于每一个松弛变量，在目标函数中对应一个代价，所以目标函数变为：

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bfav9viij30g6044t8z.jpg)



* C> 0  称为惩罚参数， C 值大，对于误分类的惩罚大。

* **若** **C ——> 0** ,误差分量实质消失，目标最大化间隔

  **若** **C ——> ∞** ,间隔消失，目标最小化误差损失

* 最小化目标函数： 间隔尽量大 + 误分类的点尽量少

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfc0chexj30jg06w3z0.jpg)

**k** **决定了误差损失的形式，**

**K = 1** ,称为 **hinge loss**    **通常选择，也是本文后面内容推导选择**

**K = 2 ,** **称为** **quadratic loss** 



#### 2. 求解参数

同样的 拉格朗日方法

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfdkwoisj30ui0eijuh.jpg)

消除 W, b 转化为

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2bfe2q9n3j30ui09udhy.jpg)

相比于上文的硬间隔分类，只是多了一个 a 的范围限制条件

还是通过 **SMO**  算法求解 参数



#### 3. 软间隔的支持向量 ------稍微复杂些

![](https://ws2.sinaimg.cn/large/006tNc79ly1fztfr5f71aj30rs0k2wic.jpg)



软间隔最大化时 KKT 条件中的对偶互补条件

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bfo60737j30g607yaaq.jpg)



1.  对于  a = 0 

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfs2pxxhj30ku06cdgf.jpg)

所以 样本点被正确的分类



2.    0< a <. C. ——— 向量为支持向量

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2bft5k8ztj30h409ggmi.jpg)

所以 推导得到

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfttnhg7j30g6044mxe.jpg)

样本点在 间隔上面，属于支持向量



3.   a = C   ——— 向量为支持向量

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bfvuoyjcj30g6082mxs.jpg)

如图所示，$\xi$  表示 点和其所属类支持向量平面的之间的距离

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bg05psfzj30t807iq4p.jpg)

或者可以用下面的条件 进行分析，来判断样本点的位置 

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bfwkluouj30g603ajrk.jpg)

**无论样本点在哪，这些点都是软间隔的支持向量，**$\xi$  表示偏差。



-----

### 3. 非线性支持向量机和 核函数



#### 1. 存在线性不可分的数据

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bg2s4wofj30bs06lab2.jpg)



观察上面的图，可以发现，对于线性不可分的数据，映射到高维空间中，就可以继续应用线性可分的 SVM思想来解决问题

**但是这样存在一个问题，初始数据维度高，再向高维映射，维度会有爆炸性的增长**** 。**而且特征空间的维度一般都很高，甚至是无穷维的，例如后文的高斯核**


#### 1. 核技巧(kernel trick)

通过非线性变换，将非线性问题变换为线性问题

核技巧应用到 SVM 中的核心思想 ： 通过一个非线性变换将输入空间转 对应到一个特征空间，使得输入空间的超曲面模型转换为 特征空间的超平面模型

**避免显式地将输入空间的每一个点映射到特征空间中，输入对象用他们之间的$n*n$ 成对相似度来表示。其中相似度函数称为** **kernel**

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bg96nfxmj30p007sta0.jpg)

> **核方法不再将数据看作** **输入空间或者特征空间的向量，而是仅仅考虑点对之间的核值。**
>
> **可以看作** **n** **个输入点的完全图带权邻接矩阵**


#### 2， 核函数定义

* X 为输入空间， H 为特征空间存在一个 X 到 H 的映射； $\phi (x)  X--> H$

* 对于 $ x,z \in X  , K(x,z) = \phi (x) * \phi(z)$

* K(x,z)  就是核函数， 使用的时候，不显示的定义 映射函数，而是直接 **依据领域知识选取核函数**

![](https://ws2.sinaimg.cn/large/006tNc79ly1g2bg60ov90j30ko040t93.jpg)

这样可以 **在特征空间中直接计算内积 〈φ(xi · φ(x)〉，**而不是，先将特征映射到高维空间，之后再在高维空间做线性分类。

> **学习是隐式的在特征空间进行，不需要显示的定义特征空间和映射函数**


#### 3. 常用核函数

* 多项式核函数 ： p次多项式分类器
* 高斯核函数 ： 高斯径向基函数分类器、
* 字符串核函数

-----



### 4. SMO 优化算法 -- 序列最小最优化算法


> 暂不详

-----

# 3. SVM 多分类

### 1.  理想情况

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2bghoz4cij30f20bb0tc.jpg)

一次多分类运算，得到多个超平面，如上图所示，分割出来的区域对应于每个类别

**存在问题** 计算复杂度太高

### 2. 一类对其余

例如有 10类，1，2，…10, 每次把一类设定为正类别，其他的全部设定为负类别，这样得到 10个 二类分类器

**存在问题**：

1. 数据集偏斜，负样本数目多
2. 分类出现重叠现象： 一个样本在第一个分类器被分为 1，第二个被分为2
3. 分类出现不可分类现象： 样本在每个二类分类器中都是负类别。



### 3. 一对一方法

每次选择一个类 的样本作为正类样本，再选择一个负类样本

例如有 4 类，二分类器 (12)，(13)，(14)，(23)，(24)，(34)，

k(k-1)/2 个分类器

测试阶段，给定一个输入，每个分类器给出一个结果，使用 VOte 机制得到最后的结果。

**存在的问题** ： 分类器的数目过多

**Sklearn 中的 SVC 就是使用这种方法**

> **The multiclass support is handled according to a one-vs-one scheme.**
>
> **If** **n_class** **is the number of classes, then** **n_class \* (n_class - 1) / 2** **classifiers are constructed and each one trains data from two classes. T**



### 4. DAG-SVM 

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bgu6dr3ij30bi0agab4.jpg)

如图，先分类判别是1，还是4 ，是1 走右边路径，再分类

树状的好处是，从根到节点最深 = 用到分类器数目= K-1,

例如左图，4个类别，对于一个新的数据，只需要判别3次

**存在的问题**： 错误的积累，一开始判别出错。

----



# 4.SVM 算法实现 — Sklearn

### 1. 实验数据

![](https://ws4.sinaimg.cn/large/006tNc79ly1fzti6zqiutj30rc0t6tcc.jpg)

* 使用 Age 和 salary 的数据 去预测是否购买 --- 二维数据便于绘制图形

### 2. Code

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

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bgd4xtulj30p205yjt1.jpg)

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0,C = 2) # 可以选择不同的 核函数
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
*  SVC kernel中参数调节 —— Gridsearch 调节

-----------

# 5. SVM用作回归问题 — SV regression

### 1. 回归问题

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bh0rdtlyj30fo061wf0.jpg)

- **对于使用** **支持向量机做** **回归预测，主要是拟合一个函数f(x)，使得所有的预测值的误差都小于** **ε**（小于就可以）
- 阴影部分叫做 margin，就是回归的范围
- **不同于一般的回归**
- **同时要使得** 拟合函数尽可能Flatness，就是 **w** **尽可能小**

### 2. 优化

类似于 SVM非分析，抽象成一个优化问题

![](https://ws3.sinaimg.cn/large/006tNc79ly1g2bh34b6foj30du056mxh.jpg)

对于软间隔 soft-margin，允许偏离部分

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bh52q53vj30ai03umxb.jpg)

### 3. 求解

拉格朗日求解

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bh5kwknij30k0098jsh.jpg)



### 4. Kernel 进行 mapping

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bh6c110zj30n00begn1.jpg)

### 5. 完整的回归步骤

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bh6u6ph0j30lo0con1h.jpg)

# 6. SVR 回归 Code — Sklearn

![image-20190422161215749](/Users/liuchuang/Library/Application Support/typora-user-images/image-20190422161215749.png)



```python
from sklearn.svm import SVR
X_train, y_train, X_test, y_test = load_dataset(filename)

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_poly) # 计算 MSE
from sklearn.metrics import mean_squared_error,mean_absolute_error
mae = mean_absolute_error(y_test, y_rbf) # 计算 MAE
```

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bhah045uj31jg0qa772.jpg)



# 7. SVR 用于多项回归

### 1. 多项回归实现原理—— sklearn

> **This strategy consists of fitting one regressor per target. Since each target is represented by exactly one regressor it is possible to gain knowledge about the target by inspecting its corresponding regressor.** 

* 每一项回归使用一个回归器，
* 没有用到数据之间的关联

### 2. Multi SVR — code

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

X_train, y_train, X_test, y_test = load_dataset2(filename)
svr_rbf_multi = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
y_rbf_multi = svr_rbf_multi.fit(X_train, y_train).predict(X_test)

def plot_result2(i,y_test,y_pred):
    
    # 绘制完整预测曲线  i 为测试集
    
    #plt.figure(figsize=(42, 20))
    plt.plot(y_pred[i,:], color='blue', lw=3, label='Predicted')
    plt.plot(y_test[i,:], color='red', lw=3, label='Raw-dose')
    plt.xlabel('Depth(mm)', fontsize='40')
    plt.ylabel('Nomalized Intensity', fontsize='40')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.text(250, 0.8, 'SNR = 3 ', fontsize='40')
    plt.legend(fontsize=30)
    #plt.show()

plt.figure(figsize=(42, 10))
for i in range(3):
    plt.subplot(1,3,i+1)
    plot_result2(i,y_test,y_rbf_multi)
```

rbf 核预测结果

![](https://ws1.sinaimg.cn/large/006tNc79ly1g2bhiou1aaj31jg0eoq8m.jpg)



-----





```python
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

svr_rbf = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
svr_lin = MultiOutputRegressor(SVR(kernel='linear', C=100, gamma='auto'))
svr_poly = MultiOutputRegressor(
    SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1))

lw = 3

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']
model_color2 = ['c', 'g', 'm']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot((svr.fit(X_train, y_train).predict(X_test))[1,:], color='r', lw=lw,
                  label='{} Real'.format(kernel_label[ix]))
    axes[ix].plot(y_test[1,:], color='b', lw=lw,label='{} Predicted'.format(kernel_label[ix]))
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True,fontsize= 20)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=20)
plt.show()

```

不同核函数预测结果

![image-20190422162121266](/Users/liuchuang/Library/Application Support/typora-user-images/image-20190422162121266.png)



----

rbf 参数调节 

```python
C_2d_range = [1e-2,1e-1, 1e1, 1e2,1e3]
gamma_2d_range = [1e-2,1e-1, 1, 1e1]
regressions = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        multi_reg = MultiOutputRegressor(
            SVR(kernel='rbf', C=C, gamma=gamma, epsilon=.1))
        multi_reg.fit(X_train, y_train)
        regressions.append((C, gamma, multi_reg))
        
plt.figure(figsize=(30, 20))
for (k, (C, gamma, multi_reg)) in enumerate(regressions):
    
    y_rbf_multi = multi_reg.predict(X_test)

    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='large')
    plot_result3(2,y_test,y_rbf_multi)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
```

调参数结果图

![](https://ws4.sinaimg.cn/large/006tNc79ly1g2bho2dc9hj30rq0i4dk1.jpg)

-----



# 附录

##### [SVM 实现的 jupyter notebook](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day11-12-13-14_SVM/SVM%20%E5%AE%9E%E7%8E%B0.ipynb)

##### [SVM 实现 python 代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day11-12-13-14_SVM/SVM_sklearn.py)

##### [数据集](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/Social_Network_Ads.csv)

SVR 的数据集没有放




# 参考

1. 统计学习方法 --- 李航
2. https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md
3. 机器学习实战
4. **刘建平** **个人博客：** [**http://www.cnblogs.com/pinard/p/6100722.html**](http://www.cnblogs.com/pinard/p/6100722.html)
5.  [**Please explain Support Vector Machines (SVM) like I am a 5 year old**](https://link.zhihu.com/?target=https%3A//www.reddit.com/r/MachineLearning/comments/15zrpp/please_explain_support_vector_machines_svm_like_i/)
6.  [**https://blog.csdn.net/qq_26898461/article/details/50481803**](https://blog.csdn.net/qq_26898461/article/details/50481803)
7. [**https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff**](https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff)
8.  [**http://kernelsvm.tripod.com/**](http://kernelsvm.tripod.com/)
9. 数据挖掘与分析， 概念与算法
10. Sklearn 官方网站
















