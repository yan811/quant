import os
import pandas as pd
import numpy as np

import sys

from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.tools as tools
import utils.calculators as calculators
import factor_gen.y as y
import mat73
from heapq import nlargest
from heapq import nsmallest
from factor_filter.factor_filter import PerformanceWithCost


def load_data_from_PX(path):
    data1 = mat73.loadmat(path)
    data1 = data1['Px']
    col = data1['LocalID']
    ind = tools.ConTimeDay(data1)
    ind = tools.Str2times(ind)

    Open = data1['Open']
    Open = pd.DataFrame(Open)
    Open.columns = col
    Open.index = ind

    filter = Open.index >= '2018-01-01'
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

    return Open[TradeDay == 1], High[TradeDay == 1], Low[TradeDay == 1], Close[TradeDay == 1], Volume[TradeDay == 1] \
        , Amount[TradeDay == 1], TotalRet[TradeDay == 1], VWAP[TradeDay == 1], TradeDay


# 布林线
def get_bollinger_band(Close, mean_day=20, std_index=2):
    mean_line = calculators.ts_Mean(Close, mean_day)
    std = calculators.ts_Stdev(Close, mean_day)
    up_line = mean_line + std_index * std
    down_line = mean_line - std_index * std
    return mean_line, up_line, down_line, std


def get_EMA(Close, num=1):
    if num == 1:
        return Close
    last_EMA = get_EMA(Close, num - 1)
    EMA = (num - 1) / (num + 1) * last_EMA.shift(1) + 2 / (num + 1) * Close
    return EMA


def get_MA(Close, num=200):
    return calculators.ts_Mean(Close, num)


def get_MACD(Close, ):
    EMA12 = get_EMA(Close, 12)
    EMA26 = get_EMA(Close, 26)
    DIF = EMA12 - EMA26
    DIF_EMA9 = get_EMA(DIF, 9)
    volume = 2 * (DIF - DIF_EMA9)
    return volume, DIF, DIF_EMA9


def get_Aroon(Close, num=20):
    eq_AroonUp = []
    eq_AroonDown = []
    for index, column in Close.iteritems():
        priceSeries = column
        eq_AroonUp.append((nlargest(1, range(len(priceSeries)), key=priceSeries.get)[0] + 1) * 100 / num)
        eq_AroonDown.append((nsmallest(1, range(len(priceSeries)), key=priceSeries.get)[0] + 1) * 100 / num)
    return eq_AroonUp


def get_EMV(Volume, High, Low, num1=14, num2=9):
    mid = calculators.ts_Delta(High, 1) + calculators.ts_Delta(Low, 1)
    bro = 2 * Volume / (High - Low)
    em = mid / bro
    EMV = calculators.ts_Mean(em, num1)
    MAEMV = calculators.ts_Mean(em, num2)
    return EMV, MAEMV


def get_CMO(Close, num):
    delta = calculators.ts_Delta(Close, 1)
    SUM = delta.rolling(window=num).sum()
    ABS_SUM = delta.abs().rolling(window=num).sum()
    return SUM / ABS_SUM * 100


def build_df_per_code(col, Close, High, Low, Open, Volume, Amount, VWAP,
                      mean_VWAP, mean_High, mean_Low, mean_Open, mean_Amount, mean_Close_5, mean_Close_10,
                      d_Amount, d_VWAP, d_Close, d_Volume, mean_line_20, std_20, f1,
                      EMV, MAEMV, d_EMV, volume, DIF_EMA9, DIF, f2):
    data = pd.DataFrame(Close[col])
    data.reset_index()
    data.columns = ['time', 'Close']
    data['High'] = High[col]
    data['Low'] = Low[col]
    data['Open'] = Open[col]
    data['Volume'] = Volume[col]
    data['Amount'] = Amount[col]
    data['VWAP'] = VWAP[col]

    data['mean_VWAP'] = mean_VWAP[col]
    data['mean_High'] = mean_High[col]
    data['mean_Low'] = mean_Low[col]
    data['mean_Open'] = mean_Open[col]
    data['mean_Amount'] = mean_Amount[col]
    data['mean_Close_5'] = mean_Close_5[col]
    data['mean_Close_10'] = mean_Close_10[col]

    data['d_Amount'] = d_Amount[col]
    data['d_VWAP'] = d_VWAP[col]
    data['d_Close'] = d_Close[col]
    data['d_Volume'] = d_Volume[col]

    data['mean_20'] = mean_line_20[col]
    data['std_20'] = std_20[col]
    data['BOLLING_factor'] = f1[col]

    data['EMV'] = EMV[col]
    data['MAEMV'] = MAEMV[col]
    data['d_EMV'] = d_EMV[col]

    data['MACD'] = volume[col]
    data['EMA'] = DIF_EMA9[col]
    data['DIF'] = DIF[col]
    data['MACD_factor'] = f2[col]

    data['code'] = int(col)
    return data.values


def make_functions():
    gp_add = make_function(function=calculators.gp_add, name='gp_add', arity=2)
    gp_sub = make_function(function=calculators.gp_sub, name='gp_sub', arity=2)
    gp_mul = make_function(function=calculators.gp_mul, name='gp_mul', arity=2)
    gp_div = make_function(function=calculators.gp_div, name='gp_div', arity=2)
    gp_sqrt = make_function(function=calculators.gp_sqrt, name='gp_sqrt', arity=1)
    gp_log = make_function(function=calculators.gp_log, name='gp_log', arity=1)
    gp_neg = make_function(function=calculators.gp_neg, name='gp_neg', arity=1)
    gp_inv = make_function(function=calculators.gp_inv, name='gp_inv', arity=1)
    gp_abs = make_function(function=calculators.gp_abs, name='gp_abs', arity=1)
    gp_sig = make_function(function=calculators.gp_sig, name='gp_sig', arity=1)
    gp_relu = make_function(function=calculators.gp_relu, name='gp_relu', arity=1)
    gp_exp = make_function(function=calculators.gp_exp, name='gp_exp', arity=1)
    gp_cos = make_function(function=calculators.gp_cos, name='gp_cos', arity=1)
    gp_sin = make_function(function=calculators.gp_sin, name='gp_sin', arity=1)
    function_set = (
        gp_add, gp_sub, gp_mul, gp_div, gp_sqrt, gp_log, gp_neg, gp_inv, gp_abs, gp_sig, gp_relu, gp_exp, gp_cos,
        gp_sin)
    return function_set


def train():
    print('generate x...')
    X = pd.read_csv('factor105.csv')
    print('generate y...')
    Y = y.get_day_rate_only_work_day()
    Y['rate'] = Y['rate'].shift(-2)  # 后移20天（用X预测20天之后的Y）
    print('X len:{}'.format(len(X)))
    print('Y len:{}'.format(len(Y)))
    assert len(X) == len(Y)
    X = X[['Close', 'High', 'Low', 'Open', 'Volume', 'Amount', 'VWAP',
           'mean_VWAP', 'mean_High', 'mean_Low', 'mean_Open', 'mean_Amount', 'mean_Close_5',
           'mean_Close_10',
           'd_Amount', 'd_VWAP', 'd_Close', 'd_Volume', 'mean_20', 'std_20', 'BOLLING_factor',
           'EMV', 'MAEMV', 'd_EMV', 'MACD', 'EMA', 'DIF', 'MACD_factor']]
    Y = Y[['rate']]
    print('train/test split...')
    split = int(len(Y) * 0.8)
    X_train = X.iloc[:split, :]
    Y_train = Y.iloc[:split, :]

    print('find functions...')
    function_set = make_functions()
    st = SymbolicTransformer(generations=30,  # 公式进化的世代数量  50
                             function_set=function_set,  # 用于构建和进化公式时使用的函数集
                             parsimony_coefficient=0.00001,  # 节俭系数，用于惩罚过于复杂的公式
                             metric='spearman',  # 适应度指标，可以用make_fitness自定义
                             p_crossover=0.9,  # 交叉变异概率
                             max_samples=0.9,  # 最大采样比例
                             verbose=1,
                             random_state=0,  # 随机数种子
                             n_jobs=-1  # 并行计算使用的核心数量
                             )

    X_train = X_train.fillna(0)
    Y_train = Y_train.fillna(0)
    st.fit(X_train, Y_train)
    df_statistics = pd.DataFrame(st.run_details_)

    print(st)
    with open('./factor105_gplearn.txt', 'w') as f:
        f.write(str(st))
    print('finish saving functions')

    # 画学习曲线
    duration = df_statistics['generation_time'].sum() / 60
    x = df_statistics['generation']
    plt.plot(x, df_statistics['average_fitness'], label='average')
    plt.plot(x, df_statistics['best_fitness'], label='best')
    plt.plot(x, df_statistics['best_oob_fitness'], label='best_oob')
    plt.title('Learning Curve of Fitness, time:%.0fmin' % duration)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('./factor105_gplearn.jpg')
    # plt.show()
    print('finish saving learning curves')


def generate_data():
    Open, High, Low, Close, Volume, Amount, TotalRet, VWAP, TradeDay \
        = load_data_from_PX('../raw_data/Px_new.mat')

    mean_VWAP = calculators.ts_Mean(VWAP, 5)
    mean_High = calculators.ts_Mean(High, 5)
    mean_Low = calculators.ts_Mean(Low, 5)
    mean_Open = calculators.ts_Mean(Open, 5)
    mean_Amount = calculators.ts_Mean(Amount, 5)
    mean_Close_5 = calculators.ts_Mean(Close, 5)
    mean_Close_10 = calculators.ts_Mean(Close, 10)

    d_Amount = calculators.ts_Delta(Amount, 1)
    d_VWAP = calculators.ts_Delta(VWAP, 1)
    d_Close = calculators.ts_Delta(Close, 1)
    d_Volume = calculators.ts_Delta(Volume, 1)

    mean_line_20, up_line_20, down_line_20, std_20 = get_bollinger_band(Close)
    f1 = (calculators.ts_Decay(calculators.ts_Delta(mean_line_20, 1), 3) * std_20) ** (
            (up_line_20 - Close) / (std_20 + 0.000001))
    EMV, MAEMV = get_EMV(Volume, High, Low)
    d_EMV = calculators.ts_Delta(EMV, 1)

    volume, DIF, DIF_EMA9 = get_MACD(Close)

    f2 = volume * calculators.ts_Delta(DIF, 1)
    columns = VWAP.columns

    total_data = []
    for col in columns:
        data = build_df_per_code(col, Close, High, Low, Open, Volume, Amount, VWAP,
                                 mean_VWAP, mean_High, mean_Low, mean_Open, mean_Amount, mean_Close_5, mean_Close_10,
                                 d_Amount, d_VWAP, d_Close, d_Volume, mean_line_20, std_20, f1,
                                 EMV, MAEMV, d_EMV, volume, DIF_EMA9, DIF, f2)
        total_data.append(data)
    total_data = pd.DataFrame(np.concatenate(total_data))
    total_data.index = Close.index ** len(Close.columns)
    total_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Amount', 'VWAP',
                          'mean_VWAP', 'mean_High', 'mean_Low', 'mean_Open', 'mean_Amount', 'mean_Close_5',
                          'mean_Close_10',
                          'd_Amount', 'd_VWAP', 'd_Close', 'd_Volume', 'mean_20', 'std_20', 'BOLLING_factor',
                          'EMV', 'MAEMV', 'd_EMV', 'MACD', 'EMA', 'DIF', 'MACD_factor']

    total_data.to_csv('./factor105.csv', encoding='utf-8')


if __name__ == '__main__':
    train()
