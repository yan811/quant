#!/usr/bin/env python
# coding: utf-8

# In[1]:


from my_utils import read_csv,get_pivot_data


# In[2]:


df = read_csv(dir_name='C:/Users/DELL/Desktop/recent/研一下/学习/量化/final_project/quant_code/raw_data/combine_data/',file_name='factor101_financial')


# In[3]:


#https://www.joinquant.com/view/community/detail/b8ae4f443d688b7d479c07dffdf3fd56?type=1
time_data = get_pivot_data(df,'time')

def get_ic_beta_t(all_data_dict,factor_name):
    """
    输入:所有截面期的因子数据字典{'2020-01-31':因子df,...}
    输出:所有因子的ic,因子收益率,t值的df
    """
    # 定义存放所有期,所有因子的ic,beta,t字典
    factors_ic_all_period={}
    factors_beta_all_period = {}
    factors_t_all_period={}

    # 遍历因子数据每一期
    for k,v in all_data_dict.items():

        # 定义存放单期所有因子的ic列表
        factors_ic_period = []
        factors_beta_period = []
        factors_t_period = []
        
        # 遍历每列因子
        for factor in v.columns[:-1]:

            ########### 因子IC ############
            # 计算与最后一列的收益秩相关系数
            ic = st.spearmanr(v[factor],v['return'])[0]
            # 依次存入列表
            factors_ic_period.append(ic)

            ########### 因子收益率,t值 ###########
            # 每列因子与收益率RLM回归,得到系数,t值
            # 加截距,变成二维
            x=sm.add_constant(v[factor])
            model = sm.RLM(v['return'],x).fit()
            factors_beta_period.append(model.params[1])
            factors_t_period.append(model.tvalues[1])
        
        # 将因子列表存入这一期的字典
        factors_ic_all_period[k] = factors_ic_period
        factors_beta_all_period[k] = factors_beta_period 
        factors_t_all_period[k] = factors_t_period
        print(f"{k}数据已处理")
    #------------将得到的3个字典转为df--------------#
    ic_df = dict_to_df(factors_ic_all_period,factor_name,'IC')
    beta_df = dict_to_df(factors_beta_all_period,factor_name,'beta')
    t_df = dict_to_df(factors_t_all_period,factor_name,'t')
    print("所有数据处理完毕!")
    return ic_df,beta_df,t_df


# In[ ]:




