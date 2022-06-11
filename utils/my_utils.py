#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import os
import tools


# In[50]:



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

def read_csvs(dir_name,write=False):
    '''
    将文件夹内的csv按照纵向合并为一个dataframe
    
    params:
    dir_name:'C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/factor101/financial/'
    write:True表示将合并的dataframe在./raw_data/combine_data/内写入csv, 文件命名eg.factor101_financial.csv
    
    return:
    dataframe
    '''
    i = 1
    for file in os.listdir(dir_name):
        code = file.replace('.csv','')
        if i==1:
            df = read_csv(dir_name,code,combine_stocks=True)
        else:
            df_temp = read_csv(dir_name,code,combine_stocks=True)
            df = pd.concat([df,df_temp],axis = 0)
        print('finish reading '+code)
        i+=1
        
    if write:
        path = 'C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/combine_data/'
        factor_type = dir_name.split('/')[-3]#'factor101'
        factor_name = dir_name.split('/')[-2]#'financial'
        df.to_csv(root+'/'+'raw_data'+'/'+'combine_data'+'/'+factor_type+'_'+factor_name+'.csv')
    return df


# In[51]:


if __name__=='__main__':
    df1 = read_csv(dir_name='C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/combine_data/',file_name='factor101_financial')
    print(df1)


# In[99]:


#待测试待优化
def get_pivot_data(df,dim_type):
    '''
    dim_type：'factor'/'code'/'time'  将想降维的维度转为字典的key
    
    三维表df[factor,time,code]转字典{factor:df[time,code])或{code:df[time,factor])或{time:df[factor,code])
    
    eg.factor
    默认 时间×因子
           因子1 因子2  code
    时间1              000001
    时间2              000001
    时间1              000002
    时间2              000002

    转为 每个因子的表
    因子1
           000001 000002
    时间1           
    时间2

    '''
    
    df['time'] = df.index
    df2 = df.reset_index(drop = 'True')
    keys = df[dim_type].values
    
    
    if dim_type=='factor':
        factor_names = df.columns[:-2]
        df_group = {}
        for key in factor_names:   
            df_temp = df2[['time',key,'code']]
            da = pd.pivot(df_temp,index = 'time',columns = 'code')
            df_group[key] = da
            
    elif dim_type=='time':
        #df_group = df2.groupby('time')
        def clean(da):
            #da = df_group.get_group(key)
            da.index = da['code']
            da = da.drop(columns=['time','code'])
            da = da.T
            return da
        df_group = df2.groupby('time').apply(lambda x:clean(x))
    elif dim_type=='code':
        #df_group = df2.groupby('code')
        def clean(da):
            #da = df_group.get_group(key)
            da.index = da['time']
            da = da.drop(columns=['time','code'])
            return da
        df_group = df2.groupby('code').apply(lambda x:clean(x))
            #da = pd.pivot(df4,index = 'time')
        #dic[key] = da
        #dic[key] = da
    return df_group

