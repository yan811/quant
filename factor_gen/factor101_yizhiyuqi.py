#!/usr/bin/env python
# coding: utf-8

# In[1]:


# change browser to chrome to open jupyter notebook : https://www.jb51.net/article/186420.htm

# import basic modules
import pandas as pd
import numpy as np
import mat73  # Comment: use " pip install mat73" in Annaconda Powershell to install mat73 
import os
#path= 'C://Users//Lenovo//lesson//'
path = './'
os.chdir(path)
import matplotlib.pyplot as plt
import tools  # local module to deel with time format change


# In[19]:


#from my_utils import read_csv
df = read_csv(dir_name = 'C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/factor101/', file_name='PerformanceForecast_DataYes')


# In[20]:


df


# In[43]:


df.columns


# In[44]:


#col = df['ID']
ind = ConTimeDay(df,'EndDate') 
ind = tools.Str2times(ind)
df['EndDate2'] =ind
df = df[df['EndDate']>=737061]#2018.1.1

df


# In[ ]:


# Calculators
import math
import scipy.stats as st


def OnlyTrading(df,TradeDay):   # turn data with all natural date to data with trade day only
    dfCleaned = df[TradeDay == 1]
    return dfCleaned

def AllDate(df,dfCleaned,TradeDay):   # opposite
    out = df.copy() 
    for col in out.columns:
        out[col].values[:] = np.nan
    out[TradeDay == 1] = dfCleaned
    return out

##  PN
def  pn_TransNorm(dfCleaned ):  # normalization
    # dfCleaned = pd.DataFrame({'a':[2.3,-1.7,5,3],'b':[6,2.9,-3.1,8],'c':[4,5.9,-6.11,8.1],'d':[7,22,-3.21,81],'e':[9,12,-1.21,11]},index=['one','two','three','four'])
    # rank
    rank1 = dfCleaned.rank(axis = 1)
    rank2  = rank1.count(1)
    rank22 = tools.Repmat(rank1,rank2)
    rank23 = tools.Repmat(rank1,1/2/rank2)
    rank4 = (rank1-1) / +  rank22
    rank5 = rank4 + rank23
    rank6 = rank5.copy()
    for v in rank6.columns:  #####
        rank6[v] =  st.norm.ppf(list(rank6[v])  )  ##normalizing 
    return rank6


## TS

def ts_Delay(df, num ):
    dfCleaned = OnlyTrading(df,TradeDay)
    dfCleaned = dfCleaned.shift(num)   # shift, like:  df.shift(1), let yesterday's data to today 
    df = AllDate(df,dfCleaned,TradeDay)
    return df    
 
    
def ts_Mean(df, num):                # equal weight
    dfCleaned = OnlyTrading(df,TradeDay)
    dfCleaned2 = dfCleaned.rolling(window=num).mean()
    df = AllDate(df,dfCleaned2,TradeDay)
    return df
    
def ts_Decay(df, num):                 # decayed weight: linear change
    dfCleaned = OnlyTrading(df,TradeDay)
    sums = 0 
    for v in range(num):
        # print(v)
        # print( (num - v ) / num)
        if v == 0:
            dfCleaned2 = dfCleaned.copy()
        else:    
            dfCleaned2 = dfCleaned2 + dfCleaned.shift(v) * (num - v ) / num
        sums = sums + (num - v ) / num
    dfCleaned2 = dfCleaned2 / sums
    df = AllDate(df,dfCleaned2,TradeDay)
    return df

def cf(j,n):
    p1 = n - j 
    p2 = n 
    p3 =  st.norm.ppf((p1 / p2))
    return p3

def ts_DecayExp(df, num):   # decayed weight: nonlinear change
    dfCleaned = OnlyTrading(df,TradeDay)
    series = [i for i in range(1, num * 2 + 1, 1)]
    out = list()
    n =len(series)
    # get weights
    for v in range(len(series)):
        j = v + 1
        out.append(cf(j,n))
    out2 = out[:num]
    sums = 0 
    dfCleaned2 = dfCleaned * 0 
    for v in range(num):
        dfCleaned2 = dfCleaned2 + dfCleaned.shift(v) * out2[v]
        sums = sums + out2[v]
    dfCleaned2 = dfCleaned2 / sums
    df = AllDate(df,dfCleaned2,TradeDay)
    return df
          
def ts_Max(df, num):                # get the max value of last num trading day
    dfCleaned = OnlyTrading(df,TradeDay)
    dfCleaned2 = dfCleaned.rolling(window=num).max()
    df = AllDate(df,dfCleaned2,TradeDay)
    return df


def ts_Min(df, num):             # get the min value of last num trading day  
    dfCleaned = OnlyTrading(df,TradeDay)
    dfCleaned2 = dfCleaned.rolling(window=num).min()
    df = AllDate(df,dfCleaned2,TradeDay)
    return df

def ts_Delta(dfCleaned, num):   
    dfCleaned2 = dfCleaned - ts_Delay(dfCleaned,num)
    return dfCleaned2


def ts_Stdev(df, num):             # get the min value of last num trading day  
    dfCleaned = OnlyTrading(df,TradeDay)
    dfCleaned2 = dfCleaned.rolling(num).std()
    df = AllDate(df,dfCleaned2,TradeDay)
    return df

# more calculator , see df.rolling: http://www.cppcns.com/jiaoben/python/301821.html 
def ts_Rank(df, num):
    dfCleaned = OnlyTrading(df,TradeDay)
    df = dfCleaned.rolling(num).rank()
    return df
#rank=gplearn.functions.make_function(function = _rank,name = 'rank',arity = 1)


def _logical(x1,x2,x3,x4):
    return np.where(x1 > x2,x3,x4)
logical = gp.functions.make_function(function = _logical,name = 'logical',arity = 4)
def _boxcox2(x1):
    with np.errstate(over='ignore', under='ignore'):
        return (np.power(x1,2)-1)/2
binary = gp.functions.make_function(function = _binary,name = 'binary',arity = 1)
function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg','inv','sin','cos','tan', 'max', 'min',boxcox2,logical]


# In[55]:


from gplearn.genetic import SymbolicTransformer
st = SymbolicTransformer(generations=5,  # 公式进化的世代数量  30
                         #function_set=function_set,  # 用于构建和进化公式时使用的函数集
                         parsimony_coefficient=0.001,  # 节俭系数，用于惩罚过于复杂的公式
                         metric='spearman', # 适应度指标，可以用make_fitness自定义
                         p_crossover=0.9,  # 交叉变异概率                     
                         max_samples=0.9,  # 最大采样比例
                         verbose=1,
                         random_state=0,  # 随机数种子
                         n_jobs=-1,  # 并行计算使用的核心数量
                        )
X=df[['EEarningRateFloor','EEarningRateCeiling', 'EEarningFloor', 'EEarningCeiling']]
y = df['EEPSCeiling']
X = X.fillna(0)
y = y.fillna(0)
st.fit(X, y)
df_statistics = pd.DataFrame(st.run_details_)

# 画学习曲线
duration = df_statistics['generation_time'].sum() / 60
x = df_statistics['generation']
plt.plot(x, df_statistics['average_fitness'], label='average')
plt.plot(x, df_statistics['best_fitness'], label='best')
plt.plot(x, df_statistics['best_oob_fitness'], label='best_oob')
plt.title('Learning Curve of Fitness, 耗时:%.0fmin' % duration)
plt.tight_layout()
plt.legend(loc='best')
plt.show()


# In[56]:


print(st)


# In[ ]:





# In[10]:


import time
def ConTimeDay(data1,col_name):
    date = data1[col_name]
    out = []
    for v in date:
        date=(v-719529)*86400+3600*8
        date = time.localtime(date)
        date = time.strftime('%Y-%m-%d',date)
        out.append(date)
    return out


# In[2]:


#已有
def read_csv(dir_name,file_name,combine_stocks=False):
    '''
    读csv
    params:
    dir_name='C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/factor101/financial/'
    file_name:文件名  合并股票是时为股票到代码 '000001.XSHE'
    combine_stocks=True为合并股票文件夹，此时将第一列（时间）设为列名，最后增加一列code

    return:
    dataframe
    '''
    path = dir_name+file_name+'.csv'
    csv_file = pd.read_csv(path,index_col = 0)
    df = pd.DataFrame(csv_file)
    
    if combine_stocks: #增加一列code       
        df['code'] = code
    #print(df)
    return df


# In[ ]:




