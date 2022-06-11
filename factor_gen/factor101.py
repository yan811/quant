#!/usr/bin/env python
# coding: utf-8

from jqdatasdk import *
import pandas as pd
auth('13120495811','Yjy19990811')#用户名 密码



def gen_data(code,start_date = '2017-12-31',end_date = '2022-06-01'):
    financial_factors = ['operating_revenue_ttm','total_profit_ttm','net_profit_ttm','operating_revenue_growth_rate','net_profit_growth_rate','roe_ttm','eps_ttm','PEG','cashflow_per_share_ttm','net_asset_per_share']
    factor_data = get_factor_values(securities=[code],factors=financial_factors,start_date=start_date, end_date=end_date)#!!!!!!!!!!!!!改code    
    i = 1
    #factor_data to df
    for key in factor_data.keys():
        #print(key,i)
        if i ==1:
            df = factor_data[key]
            df.columns = [key]
        else:
            df[key] = factor_data[key].values   
        i+=1
    df.to_csv(path+code+'.csv')
    print('finish saving '+code)

    


# In[37]:


if __name__=='__main__':
    codes = list(get_all_securities(['stock']).index)
    path = './raw_data/factor101/financial/'
    start_index = codes.index('000059.XSHE')
    for code in codes[start_index+1:]:
        gen_data(code)


# In[ ]:




