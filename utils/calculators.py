#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Calculators
import math
import scipy.stats as st
import numpy as np
from .tools import Repmat


def OnlyTrading(df, TradeDay):  # turn data with all natural date to data with trade day only
    dfCleaned = df[TradeDay == 1]
    return dfCleaned


def AllDate(df, dfCleaned, TradeDay):  # opposite
    out = df.copy()
    for col in out.columns:
        out[col].values[:] = np.nan
    out[TradeDay == 1] = dfCleaned
    return out


##  PN
def pn_TransNorm(dfCleaned):  # normalization
    # dfCleaned = pd.DataFrame({'a':[2.3,-1.7,5,3],'b':[6,2.9,-3.1,8],'c':[4,5.9,-6.11,8.1],'d':[7,22,-3.21,81],'e':[9,12,-1.21,11]},index=['one','two','three','four'])
    # rank
    rank1 = dfCleaned.rank(axis=1)
    rank2 = rank1.count(1)
    rank22 = Repmat(rank1, rank2)
    rank23 = Repmat(rank1, 1 / 2 / rank2)
    rank4 = (rank1 - 1) / +  rank22
    rank5 = rank4 + rank23
    rank6 = rank5.copy()
    for v in rank6.columns:  #####
        rank6[v] = st.norm.ppf(list(rank6[v]))  ##normalizing
    return rank6


## TS
def ts_Delay(dfCleaned, num):
    dfCleaned = dfCleaned.shift(num)  # shift, like:  df.shift(1), let yesterday's data to today
    return dfCleaned


def ts_Mean(dfCleaned, num):  # equal weight
    dfCleaned2 = dfCleaned.rolling(window=num).mean()
    return dfCleaned2


def ts_Decay(dfCleaned, num):  # decayed weight: linear change
    sums = 0
    for v in range(num):
        # print(v)
        # print( (num - v ) / num)
        if v == 0:
            dfCleaned2 = dfCleaned.copy()
        else:
            dfCleaned2 = dfCleaned2 + dfCleaned.shift(v) * (num - v) / num
        sums = sums + (num - v) / num
    dfCleaned2 = dfCleaned2 / sums
    return dfCleaned2


def ts_Decay2(dfCleaned, num):  # decayed weight: linear change
    num = min(num, len(dfCleaned))
    sums = 0
    for v in range(num):
        # print(v)
        # print( (num - v ) / num)
        if v == 0:
            dfCleaned2 = dfCleaned.copy()
        else:
            dfCleaned2 = dfCleaned2 + dfCleaned.shift(v) * (num - v) / num
        sums = sums + (num - v) / num
    dfCleaned2 = dfCleaned2 / sums
    return dfCleaned2


def cf(j, n):
    p1 = n - j
    p2 = n
    p3 = st.norm.ppf((p1 / p2))
    return p3


def ts_DecayExp(dfCleaned, num):  # decayed weight: nonlinear change
    series = [i for i in range(1, num * 2 + 1, 1)]
    out = list()
    n = len(series)
    # get weights
    for v in range(len(series)):
        j = v + 1
        out.append(cf(j, n))
    out2 = out[:num]
    sums = 0
    dfCleaned2 = dfCleaned * 0
    for v in range(num):
        dfCleaned2 = dfCleaned2 + dfCleaned.shift(v) * out2[v]
        sums = sums + out2[v]
    dfCleaned2 = dfCleaned2 / sums
    return dfCleaned2


def ts_Max(dfCleaned, num):  # get the max value of last num trading day
    dfCleaned2 = dfCleaned.rolling(window=num).max()
    return dfCleaned2


def ts_Min(dfCleaned, num):  # get the min value of last num trading day
    dfCleaned2 = dfCleaned.rolling(window=num).min()
    return dfCleaned2


def ts_Delta(dfCleaned, num):
    dfCleaned2 = dfCleaned - ts_Delay(dfCleaned, num)
    return dfCleaned2


def ts_Stdev(dfCleaned, num):  # get the min value of last num trading day
    dfCleaned2 = dfCleaned.rolling(num).std()
    return dfCleaned2


# more calculator , see df.rolling: http://www.cppcns.com/jiaoben/python/301821.html 
def ts_Rank(dfCleaned, num):
    df = dfCleaned.rolling(num).rank()
    return df


# rank=gplearn.functions.make_function(function = _rank,name = 'rank',arity = 1)


def _protected_division(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 1e-10, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-10, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-10, 1. / x1, 0.)


def _sigmoid(x1):
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def gp_add(x, y):
    return x + y


def gp_sub(x, y):
    return x - y


def gp_mul(x, y):
    return x * y


def gp_div(x, y):
    return _protected_division(x, y)


def gp_sqrt(data):
    return _protected_sqrt(data)


def gp_log(data):
    return _protected_log(data)


def gp_neg(data):
    return np.negative(data)


def gp_inv(data):  # 倒数

    return _protected_inverse(data)


def gp_abs(data):
    return np.abs(data)


def gp_sig(data):
    return _sigmoid(data)
