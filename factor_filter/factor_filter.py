#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.my_utils import read_csv, get_pivot_data

# In[2]:


df = read_csv(dir_name='C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/combine_data/',
              file_name='factor101_financial')

# In[3]:


# https://www.joinquant.com/view/community/detail/b8ae4f443d688b7d479c07dffdf3fd56?type=1
time_data = get_pivot_data(df, 'time')


def get_ic_beta_t(all_data_dict, factor_name):
    """
    输入:所有截面期的因子数据字典{'2020-01-31':因子df,...}
    输出:所有因子的ic,因子收益率,t值的df
    """
    # 定义存放所有期,所有因子的ic,beta,t字典
    factors_ic_all_period = {}
    factors_beta_all_period = {}
    factors_t_all_period = {}

    # 遍历因子数据每一期
    for k, v in all_data_dict.items():

        # 定义存放单期所有因子的ic列表
        factors_ic_period = []
        factors_beta_period = []
        factors_t_period = []

        # 遍历每列因子
        for factor in v.columns[:-1]:
            ########### 因子IC ############
            # 计算与最后一列的收益秩相关系数
            ic = st.spearmanr(v[factor], v['return'])[0]
            # 依次存入列表
            factors_ic_period.append(ic)

            ########### 因子收益率,t值 ###########
            # 每列因子与收益率RLM回归,得到系数,t值
            # 加截距,变成二维
            x = sm.add_constant(v[factor])
            model = sm.RLM(v['return'], x).fit()
            factors_beta_period.append(model.params[1])
            factors_t_period.append(model.tvalues[1])

        # 将因子列表存入这一期的字典
        factors_ic_all_period[k] = factors_ic_period
        factors_beta_all_period[k] = factors_beta_period
        factors_t_all_period[k] = factors_t_period
        print(f"{k}数据已处理")
    # ------------将得到的3个字典转为df--------------#
    ic_df = dict_to_df(factors_ic_all_period, factor_name, 'IC')
    beta_df = dict_to_df(factors_beta_all_period, factor_name, 'beta')
    t_df = dict_to_df(factors_t_all_period, factor_name, 't')
    print("所有数据处理完毕!")
    return ic_df, beta_df, t_df


# In[ ]:


def GetCost(f1_stand_D2, cost):
    Cost = cost * (f1_stand_D2 - ts_Delay(f1_stand_D2, 1))
    return Cost.abs()


def GetTurnover(f1_stand):
    a = (f1_stand - ts_Delay(f1_stand, 1)).abs().sum(1)
    b = ts_Delay(f1_stand, 1).abs().sum(1)
    b[b == 0] = np.nan
    z = a / b
    c = round(z.mean(), 3)
    return c


def PerformanceWithCost(f1, TotalRet, delayNum, cost, fig, SDate, EDate):
    # 1 normalize
    f1_stand = pn_TransNorm(f1.round(4))
    # 2 ts_Delay 2 days
    f1_stand_D2 = ts_Delay(f1_stand, delayNum)
    # get the factor return for each stocks
    factorRet = f1_stand_D2 * TotalRet
    # 3 convert negative factor to positive
    if factorRet.mean(1).mean() < 0:
        factorRet = factorRet * -1
    # 4 get cost
    Cost = GetCost(f1_stand_D2, cost)
    factorRet = factorRet - Cost
    # 5 choose date & clean data
    factorRet = factorRet.iloc[SDate:EDate]
    factorRetLine = factorRet.mean(1)
    factorRetLine[factorRetLine == 0] = np.nan
    # 6 get Sharpe ratio and annual return
    sr1 = factorRetLine.mean() / factorRetLine.std() * 15
    ret1 = factorRetLine.mean() * 250
    # 7 plot picture if fig == 1
    if fig == 1:
        factorRetLine[np.isnan(factorRetLine)] = 0
        Cumsums = factorRetLine.cumsum()
        plt.plot(Cumsums)
        plt.title('Factor Return: Sharpe Ratio = ' + str(round(sr1, 4)))
    # print
    print('SR:', round(sr1, 3), 'AR:', round(ret1, 3), 'TO:', GetTurnover(f1_stand))
    return round(sr1, 3), round(ret1, 3), factorRetLine


def CorrValue(f1, f2):
    f1_ = f1.copy()
    f2_ = f2.copy()
    f2_[f2_.isna()] = 0
    f1_[f1_.isna()] = 0
    corrs = list()
    for v in f1_.index:
        cor_ = np.corrcoef(f1_[f1_.index == v].values, f2_[f2_.index == v].values)
        cor_ = cor_[0, 1]
        corrs.append(cor_)
    return round(np.nanmean(corrs), 4)


def CorrRet(ret1, ret2):
    idx = ret1 != 0
    ret1_ = ret1[idx]
    ret2_ = ret2[idx]
    cor_ = np.corrcoef(ret1_, ret2_)
    cor_ = cor_[0, 1]
    return round(cor_, 4)
