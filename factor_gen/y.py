#!/usr/bin/env python
# coding: utf-8

# In[1]:


from jqdatasdk import *
import pandas as pd
auth('13120495811','Yjy19990811')


# In[2]:



#print(len(codes))#4864


# In[31]:

if __name__=='__main__':
    path = 'C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/y/'
    codes = list(get_all_securities(['stock']).index)
    price = get_price(security=codes,start_date='2017-12-31', end_date='2022-06-01',frequency = 'daily',fields = 'close',skip_paused=False, fq='pre')#!!!!!!!!!!!!!改code
    print(price)
'''
def gen_close_price(codes,start_date = '2017-12-31',end_date = '2022-06-01',save = False):
    price = get_price(security=codes,start_date=start_date, end_date=end_date,frequency = 'daily',fields = 'close',skip_paused=False, fq='pre')#!!!!!!!!!!!!!改code    
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
        if
    df.to_csv(path+'close_price_daily'+'.csv')
    print('finish saving ')

    


codes = list(get_all_securities(['stock']).index)

gen_data(codes)


# In[ ]:
'''




