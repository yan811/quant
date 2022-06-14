# quant

**environment**
conda create --name quant python=3.7
conda activate quant

pip install mat73,gplearn,matplotlib


2018.1.1-2022.6.1，全部A股

**structure**

| factor_filter  筛选因子&股票
--| factor_filter.py 筛因子（1、相关系数  2、夏普）
--| stock_filter.py 筛股票（1、去除ST,*ST  2、...）
| raw_data 存所有原始数据，一级子文件夹以因子编号命名（如factor101），二级命名随意，但factor_gen下相关的代码要与此处命名保持一致
----| combine_data 存所有初步清洗好的数据，二级文件夹合并成csv，命名格式factor101_×××

----| factor101

| factor_gen 放获取数据&调基本数据格式的代码，数据格式包括：时间%Y-%m-%d；股票代码格式；存储格式参考raw_data/combine_data/factor101_financial.csv...

| utils
--| calculators.py 放所有计算类函数
--|my_utils.py 放所有工具类函数（如读写数据、三维变二维...）
--|tools.py