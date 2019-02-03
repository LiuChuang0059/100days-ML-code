
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#数据预处理" data-toc-modified-id="数据预处理-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据预处理</a></span></li><li><span><a href="#特征归一化" data-toc-modified-id="特征归一化-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>特征归一化</a></span></li><li><span><a href="#SVM-的线性kernel-进行训练" data-toc-modified-id="SVM-的线性kernel-进行训练-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>SVM 的线性kernel 进行训练</a></span></li><li><span><a href="#训练集效果" data-toc-modified-id="训练集效果-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>训练集效果</a></span></li><li><span><a href="#测试集效果" data-toc-modified-id="测试集效果-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>测试集效果</a></span></li></ul></div>

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import DataFrame


# ## 数据预处理

# In[6]:


dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = DataFrame(dataset)
dataset


# In[7]:


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# In[11]:


from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split 将不再使用
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## 特征归一化

# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ##  SVM 的线性kernel 进行训练

# In[25]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, C = 2)
classifier.fit(X_train, y_train)


# In[26]:


y_pred = classifier.predict(X_test)


# ## 训练集效果

# In[27]:


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


# ## 测试集效果

# In[28]:


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

