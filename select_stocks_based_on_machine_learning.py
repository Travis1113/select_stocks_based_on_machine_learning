#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:35:38 2020

@author: gupei
"""


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("工作簿1.csv",index_col = '证券代码')

data=data.replace('——', '')
data1=data.drop('RSI相对强弱指标\n[交易日期]3月前\n[周期]14\n[复权方式]不复权\n[计算周期]日',axis=1)
data2=data['RSI相对强弱指标\n[交易日期]3月前\n[周期]14\n[复权方式]不复权\n[计算周期]日'].to_frame()
for i in range(len(data1.iloc[:,1])):
    for j in range(len(data1.iloc[1,:])):
        if ',' in data1.iloc[i,j]:
            data1.iloc[i,j]=data1.iloc[i,j].replace(',','')

data1=data1.replace('', np.nan)
data1=data1.astype(float)
data=pd.merge(data1,data2,left_index=True,right_index=True)
data=data.dropna()

data['收益率']=(data['月收盘价\n[交易日期]2020-04-17\n[复权方式]不复权']-data['月收盘价\n[交易日期]2020-03-19\n[复权方式]不复权'])/data['月收盘价\n[交易日期]2020-03-19\n[复权方式]不复权']

df=pd.DataFrame(index=data.index)
def sorting(columns,data):
    data=data.sort_index(by=columns, ascending=False)
    data['排名']=[i for i in range(len(data))]
    data['New']=data['排名']/len(data)
    data=data.sort_index()
    return data['New']

for column in ['市盈率(PE,TTM)\n[交易日期]3月前', '市销率(PS，TTM)\n[交易日期]3月前',
       '净资产收益率ROE(TTM)\n[交易日期]3月前\n[单位]%\n[TTM基准日]报表公告日期',
       '总资产报酬率ROA(TTM)\n[交易日期]3月前\n[单位]%\n[TTM基准日]报表公告日期',
       '企业倍数(EV2/EBITDA)\n[交易日期]3月前',
       'BIAS乖离率\n[交易日期]3月前\n[周期]20\n[复权方式]不复权\n[计算周期]日\n[单位]%',
       '换手率\n[交易日期]3月前\n[单位]%',
       '营业利润/营业总收入(TTM)\n[交易日期]3月前\n[单位]%\n[TTM基准日]报表公告日期',
       'RSI相对强弱指标\n[交易日期]3月前\n[周期]14\n[复权方式]不复权\n[计算周期]日']:
    df[column]=sorting(column,data)
df['收益率']=data['收益率']
def best_worst_return(rate,df):
    cd=[]
    df=df.sort_index(by='收益率',ascending=False)
    for num in range(int(len(df)*rate),len(df)-int(len(df)*rate)):
        cd.append(num)
    new_df=df.drop(df.index[cd])
    return new_df
df=best_worst_return(0.3,df)

df['收益率']=df['收益率']*100
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,0:9],df['收益率'],random_state=0)
X_test=X_test*100
X_test=X_test.astype(int)
X_train=X_train*100
X_train=X_train.astype(int)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)
grid_search.fit(X_train,y_train.astype(int) )
grid_search.score(X_test,y_test.astype(int) )






















