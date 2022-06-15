#!/usr/bin/env python
# coding: utf-8


# change browser to chrome to open jupyter notebook : https://www.jb51.net/article/186420.htm

# import basic modules
import pandas as pd
import numpy as np

import mat73  # Comment: use " pip install mat73" in Annaconda Powershell to install mat73 

import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '..')#若在ipynb中，env_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), '..')
if env_path not in sys.path:
    sys.path.append(env_path)\

import utils.tools as tools # local module to deel with time format change
import utils.calculators as calculators
import factor_gen.factor101

root= './'

def get_day_rate():
    file = os.path.join(root,'raw_data','Px_new.mat')
    data1 = mat73.loadmat(file)
    data1 = data1['Px']

    col = data1['LocalID']
    ind = tools.ConTimeDay(data1) 
    ind = tools.Str2times(ind)

    Close = data1['AdjClose']
    Close = pd.DataFrame(Close)
    Close.columns  = col
    Close.index = ind
    
    Rate = Close/Close.shift(1)-1#收盘价/昨日收盘价-1
    
    Rate = Rate[Rate.index>='2018-01-01']

    del data1,Close
    
    #拼接
    i = 1
    for code in col:
        if i ==1:
            data = pd.DataFrame(Rate[code])
            data = data.reset_index()
            data.columns = ['time','rate']
            data ['code'] = int(code)
        else:
            data_temp = pd.DataFrame(Rate[code])
            data_temp = data_temp.reset_index()
            data_temp.columns = ['time','rate']
            data_temp ['code'] = int(code)
            data = data.append(data_temp)
            del data_temp
        if i%500 == 0: print(i)
        i+=1
    return data



#rate = get_dat_rate()
def get_y(rate,percent):
    def apply_range(x,percent = percent):
        if x>percent:
            x=1
        if x<-percent:
            x=-1
        if x>=-percent and x<=percent:
            x=0
        return x
    rate['y'] = rate['rate'].apply(lambda x:apply_range(x))
    rate = rate.drop(columns = ['rate'])
    return rate



if __name__=='__main__':
    rate = get_day_rate()#收益率
    y = get_y(rate,percent = 0.03)#涨跌类别
    print(y.groupby('y').count())

