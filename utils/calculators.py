#!/usr/bin/env python
# coding: utf-8

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

def ts_Decay2(dfCleaned, num):                 # decayed weight: linear change
    num = min(num,len(dfCleaned))
    #dfCleaned = OnlyTrading(df,TradeDay)
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
    #df = AllDate(df,dfCleaned2,TradeDay)
    return dfCleaned2

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

