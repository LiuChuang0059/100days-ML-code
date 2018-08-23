# KNN 算法

> KNN基础学习详见--> https://github.com/LiuChuang0059/Machine_Learning/blob/master/Statical_Learning/Chapter_3-KNN/README.md


-----------------------
-----------------------


![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/Day%207.jpg)



## KNN 算法工作简介

* 主要： 对未标记的对象进行标记
* 计算实例点与标记的对象之间的距离，确定其k近邻点
* 使用周边数量最多的类标签来确定该对象的标签


## KNN算法实现

### 1 数据
  
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/knn%E6%95%B0%E6%8D%AE.png)


-------------------
-------------------



### 2 算法实现

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
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

```


#### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

#### Fitting K-NN to the Training set
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

```

#### Predicting the Test set results
```python

y_pred = classifier.predict(X_test)
```


#### visualisation-------详见--->[Lpgistic_regression](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day4-5-6_Logistic_regression/README.md#step-5-visulisation)

```python
plot_decision_regions(X_test, y_pred, classifier=cl)
plt.title("Test set")
```

## 结果


![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/knn_trainingset.png)
![](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/knn-testSet.png)


-----------------
------------------

# 附

### [完整代码](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/KNN.py)

### [数据集](https://github.com/LiuChuang0059/100days-ML-code/blob/master/Day7-8-9-10KNN/Social_Network_Ads.csv)



# 问题
- [ ] 尝试自实现knn分类算法

- [ ] 改变调整参数---效果




# 参考
KNeighborsClassfier参数：
http://sklearn.apachecn.org/cn/0.19.0/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

KNeighborsClassfier使用 ： http://sklearn.apachecn.org/cn/0.19.0/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py


KNN 算法python实现 ： https://www.cnblogs.com/ybjourney/p/4702562.html



















