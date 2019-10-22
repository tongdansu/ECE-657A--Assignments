#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import itertools
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
data=pd.read_csv('Desktop/ece657/breast-cancer-wisconsin.csv',header=None,names=['id','feature_1',"feature_2",'feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9','class'])
data1=data.astype(str)
delet=data1[~data1['feature_6'].str.contains("\?")]
data2=delet.astype(int)
data2.drop_duplicates()
data3=data2.drop(['id'],axis=1)
data3=data3.drop(['class'],axis=1)

def describ(x):
    print ("Mean:\n",x.mean())
    print("Mode:\n",x.mode())
    print("Skew:\n",x.skew())
    print("Standard Deviation:\n",x.std())
    print("Variance Values:\n",x.var())
df=describ(data3)

def PCC(x):
    correlations = {}
    columns = x.columns.tolist()

    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(x.loc[:, col_a], x.loc[:, col_b])

    result = DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    print(result.sort_index())
    return result
df1=PCC(data3)
df_s=df1[df1['PCC']>0.5]
print("\n For those features that PCC > 0.5, they have strong positive correlation: \n")
print (df_s)
df_w=df1[df1['PCC']<0.5]
print("\n For those features that 0< PCC < 0.5, they have weak positive correlation: \n")
print (df_w)
data_m=data2[~data2['class'].isin([2])]
data_b=data2[~data2['class'].isin([4])]
***this***
def plot(x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x['feature_1'])
    plt.title('Class M')
    plt.xlabel('Feature_1 Diagnosis Result')
    plt.ylabel('Numbers of Occurences')
    plt.show()
plot(data_m)

def plot(x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x['feature_1'])
    plt.title('Class B')
    plt.xlabel('Feature_1 Diagnosis Result')
    plt.ylabel('Numbers of Occurences')
    plt.show()
plot(data_b)


# In[ ]:




