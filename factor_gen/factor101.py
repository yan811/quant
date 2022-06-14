#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
root= 'C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/'
os.chdir(root)
import utils.tools as tools
import utils.calculators as calculators
import factor_gen.y as y


# In[2]:


def transfer_data(sample):
    sample2 = sample.set_index(['PubDate'])
    sample2.index = pd.to_datetime(sample2.index)
    sample2 = sample2[~sample2.index.duplicated()]#去除重复索引
    sample2 = sample2.resample('D').asfreq().ffill()#对齐到每天，若对齐5分钟-5T
    sample2 = sample2.reset_index()
    return sample2


# In[3]:


def get_raw_factor101():
    '''预测数据'''
    forecast_data = pd.read_csv(os.path.join(root,'raw_data','factor101','PerformanceForecast_DataYes.csv'))
    forecast_data = forecast_data.drop(columns = ['ID','RptDate','UPDATE_TIME','EndDate'])

    #处理forecast object
    dic = {'A':1, 'Q1':2, 'Q3':3, 'S1':4}
    forecast_data['ForecastObject'] = forecast_data['ForecastObject'].apply(lambda x: dic[x])

    forecast_data = forecast_data[forecast_data['PubDate']>=737061]#取时间窗  737061：2018.1.1
    forecast_data['PubDate'] = tools.Str2times(tools.ConTimeDay2(forecast_data['PubDate']))
    
    #时间对齐到日频
    forecast_data = forecast_data.groupby('Ticker').apply(lambda x:transfer_data(x))
    
    #重置索引
    forecast_data = forecast_data.reset_index(drop =True)
    print('finish preparing forecast data')
    
    '''资产负债表'''
    balancesheet_data = pd.read_csv(os.path.join(root,'raw_data','factor101','FinBalanceSheet_DataYes.csv'))

    equity = balancesheet_data[['Ticker','EndDate','PublDate','T_SH_EQUITY']]#归母净利润/总股本
    print('finish preparing balance sheet data')
    
    '''损益表'''
    incomestatement_data = pd.read_csv(os.path.join(root,'raw_data','factor101','FinIncomeStat_DataYes.csv'))
    incomestatement_data = incomestatement_data[['Ticker','EndDate','PublDate','REVENUE','T_PROFIT','DILUTED_EPS','INCOME_TAX']]
    #计算净利润
    incomestatement_data['N_PROFIT'] = incomestatement_data['T_PROFIT'] - incomestatement_data['INCOME_TAX']
    incomestatement_data.drop(columns = 'INCOME_TAX')
    #计算增长率部分
    incomestatement_data = incomestatement_data.sort_values(by =['Ticker','EndDate'],ascending =False)
    #营业收入增长率
    incomestatement_data['REVENUE_GROWTH'] = incomestatement_data['REVENUE']/incomestatement_data['REVENUE'].shift(1)
    #利润增长率
    profit_pre = incomestatement_data['T_PROFIT'].drop(0,axis = 0)
    profit_pre = np.append(np.array(profit_pre),1)
    incomestatement_data['T_PROFIT_GROWTH'] = incomestatement_data['T_PROFIT']/incomestatement_data['T_PROFIT'].shift(1)
    print('finish preparing incomestatement data')
    
    '''并表'''
    '''合并两张财务报表'''
    combine_data = pd.merge(incomestatement_data,equity,how = 'outer',on = ['Ticker','EndDate'])
    #计算ROE
    combine_data['ROE'] = combine_data['T_PROFIT']/combine_data['T_SH_EQUITY']
    combine_data = combine_data.drop(columns = 'T_SH_EQUITY')
    #保留早的PubDate
    combine_data['PubDate'] = combine_data[['PublDate_x', 'PublDate_y']].min(axis=1)
    combine_data = combine_data.drop(columns = ['PublDate_x', 'PublDate_y','EndDate'])
    #处理时间
    combine_data = combine_data[combine_data['PubDate']>=737061]#取时间窗  737061：2018.1.1
    combine_data['PubDate'] = tools.Str2times(tools.ConTimeDay2(combine_data['PubDate']))
    #时间对齐到日频
    combine_data = combine_data.groupby('Ticker').apply(lambda x:transfer_data(x))
    combine_data = combine_data.reset_index(drop =True)
    
    '''并入预期数据'''
    combine_data2 = pd.merge(combine_data,forecast_data,how = 'outer',on = ['Ticker','PubDate']) #combine_data-pubdate_x，forecast_data-pubdate_y
    #按报告期正序排序
    #combine_data2 = combine_data2.sort_values(by =['EndDate'],ascending =True)
    print('finish preparing 3 datas')
    
    return combine_data2


# In[14]:


data = get_raw_factor101()


# In[15]:


data


# In[19]:


data[list(data.columns)[2:]] = data.groupby('Ticker').apply(lambda x:calculators.ts_Decay2(x[list(x.columns)[2:]],5)).reset_index(drop =True)


# In[23]:


data['Ticker'] = data['Ticker'].astype(int)


# In[21]:


y = y.get_day_rate()


# In[22]:


y


# In[24]:


total_data = pd.merge(data,y,how = 'inner',left_on = ['Ticker','PubDate'],right_on = ['code','time'])


# In[25]:


total_data


# In[31]:


total_data = total_data.sort_values(by =['PubDate'],ascending =True)


# In[32]:


total_data


# In[34]:


X = total_data[['REVENUE',
 'T_PROFIT',
 'DILUTED_EPS',
 'INCOME_TAX',
 'N_PROFIT',
 'REVENUE_GROWTH',
 'T_PROFIT_GROWTH',
 'ROE',
 'ForcastType',
 'FiscalPeriod',
 'ForecastObject',
 'EEarningRateFloor',
 'EEarningRateCeiling',
 'EEarningFloor',
 'EEarningCeiling',
 'EProfitRateFloor',
 'EProfitRateCeiling',
 'EProfitFloor',
 'EProfitCeiling',
 'EProfitParentRateFloor',
 'EProfitParentRateCeiling',
 'EProfitParentFloor',
 'EProfitParentCeiling',
 'EEPSFloor',
 'EEPSCeiling']]
Y = total_data[['rate']]


# In[38]:


split = int(len(total_data)*0.7)
X_train = X.iloc[:,:split]
X_test = X.iloc[:,split:]
Y_train = Y.iloc[:,:split]
Y_test = Y.iloc[:,split:]


# In[ ]:


from gplearn.genetic import SymbolicTransformer
st = SymbolicTransformer(generations=50,  # 公式进化的世代数量  30
                         #function_set=function_set,  # 用于构建和进化公式时使用的函数集
                         parsimony_coefficient=0.001,  # 节俭系数，用于惩罚过于复杂的公式
                         metric='spearman', # 适应度指标，可以用make_fitness自定义
                         p_crossover=0.9,  # 交叉变异概率                     
                         max_samples=0.9,  # 最大采样比例
                         verbose=1,
                         random_state=0,  # 随机数种子
                         n_jobs=-1,  # 并行计算使用的核心数量
                        )

X_train = X_train.fillna(0)
Y_train = Y_train.fillna(0)
st.fit(X_train, Y_train)
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

print(st)


# In[ ]:





# In[ ]:




