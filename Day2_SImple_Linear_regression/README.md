# 简单线性回归

![](https://github.com/LiuChuang0059/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%202.jpg)


## 1 goal

### 1 使用单一特征 预测 结果（response）
* 假定两个变量线性相关，尝试寻找一个线性方程 来进行尽可能精确的预测

### 2寻找最好的拟合直线
* 最小化 实际观测值$y_{i}$和预测值$y_{p}$之间的长度
$$min{sum(y_{i}-y_{p})^{2}}$$
