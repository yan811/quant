import os
import pandas as pd
import numpy as np
import utils.tools as tools
import utils.calculators as calculators
import factor_gen.y as y


def load_data_from_PX(path):
    cut = 5000  # Because memory or CPU is not enough： cut the early year data

    data1 = mat73.loadmat(path)

    col = data1['LocalID']
    ind = tools.ConTimeDay(data1)
    ind = tools.Str2times(ind)
    filter = ind >= '2018-01-01'

    Open = data1['Open']
    Open = pd.DataFrame(Open)
    Open.columns = col
    Open.index = ind
    Open = Open.iloc[filter]

    High = data1['High']
    High = pd.DataFrame(High)
    High.columns = col
    High.index = ind
    High = High.iloc[filter]

    Low = data1['Low']
    Low = pd.DataFrame(Low)
    Low.columns = col
    Low.index = ind
    Low = Low.iloc[filter]

    Close = data1['Close']
    Close = pd.DataFrame(Close)
    Close.columns = col
    Close.index = ind
    Close = Close.iloc[filter]

    Volume = data1['Volume']  # 日交易量
    Volume = pd.DataFrame(Volume)
    Volume.columns = col
    Volume.index = ind
    Volume = Volume.iloc[filter]

    Amount = data1['Value']  # 日交易额
    Amount = pd.DataFrame(Amount)
    Amount.columns = col
    Amount.index = ind
    Amount = Amount.iloc[filter]

    TotalRet = data1['TotalRet']
    TotalRet = pd.DataFrame(TotalRet)
    TotalRet.columns = col
    TotalRet.index = ind
    TotalRet = TotalRet.iloc[filter]

    VWAP = data1['VWAP']
    VWAP = pd.DataFrame(VWAP)
    VWAP.columns = col
    VWAP.index = ind
    VWAP = VWAP.iloc[filter]


    # 清除新上市的股票
    listed = Volume.copy()
    listed[listed.isna()] = 0
    listed = listed.cumsum()  # get the cumsum values
    listed = listed.shift(30)
    listed[listed.isna()] = 0

    TotalRet[listed == 0] = np.nan
    Volume[listed == 0] = np.nan
    Amount[listed == 0] = np.nan
    Open[listed == 0] = np.nan
    High[listed == 0] = np.nan
    Low[listed == 0] = np.nan
    Close[listed == 0] = np.nan
    VWAP[listed == 0] = np.nan

    TradeDay = Amount.sum(1)
    TradeDay[TradeDay > 0] = 1

    return Open, High, Low, Close, Volume, Amount, TotalRet, VWAP, TradeDay


def get_bollinger_band(Close, mean_day=20, std_index=2):
    mean_line = calculators.ts_Mean(Close, mean_day)
    std = calculators.st




if __name__ == '__main__':
    Open, High, Low, Close, Volume, Amount, TotalRet, VWAP, TradeDay\
        = load_data_from_PX('./raw_data/Px_new.mat')
