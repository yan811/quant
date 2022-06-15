import os
import pandas as pd
import numpy as np
import utils.tools as tools
import utils.calculators as calculators
import factor_gen.y as y


def load_data_from_PX(path):
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


# 布林线
def get_bollinger_band(Close, TradeDay, mean_day=20, std_index=2):
    mean_line = calculators.ts_Mean(Close, mean_day, TradeDay)
    std = calculators.ts_Stdev(Close, mean_day, TradeDay)
    up_line = mean_line + std_index * std
    down_line = mean_line - std_index * std
    return mean_line, up_line, down_line


def get_EMA(Close, num=1):
    if num == 1:
        return Close
    last_EMA = get_EMA(Close, num - 1)
    EMA = (num - 1) / (num + 1) * last_EMA.shift(1) + 2 / (num + 1) * Close
    return EMA


def get_MA(Close, TradeDay, num=200):
    return calculators.ts_Mean(Close, num, TradeDay)


def get_MACD(Close, TradeDay, ):
    Close = calculators.OnlyTrading(Close, TradeDay)
    EMA12 = get_EMA(Close, 12)
    EMA26 = get_EMA(Close, 26)
    DIF = EMA12 - EMA26
    DIF_EMA9 = get_EMA(DIF, 9)
    volume = 2 * (DIF - DIF_EMA9)
    return volume, DIF, DIF_EMA9


def get_Aroon(Close, TradeDay, num=20):
    Close = calculators.OnlyTrading(Close, TradeDay)
    max_index = Close.rolling(windows=num).argmax()
    min_index = Close.rolling(windows=num).argmin()
    index = np.linspace(0, len(Close.index)).repeat(len(Close.columns), axis=0).T
    Aroon_up = (- (max_index - index)).round(num) / num
    Aroon_down = (- (min_index - index)).round(num) / num
    Aroon_osc = Aroon_up - Aroon_down
    return Aroon_up, Aroon_down, Aroon_osc


if __name__ == '__main__':
    Open, High, Low, Close, Volume, Amount, TotalRet, VWAP, TradeDay \
        = load_data_from_PX('./raw_data/Px_new.mat')
