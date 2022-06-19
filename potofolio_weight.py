import scipy.optimize
from pandas import *
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import mat73  # Comment: use " pip install mat73" in Annaconda Powershell to install mat73 
import os

root = './'
os.chdir(root)

import utils.tools as tools # local module to deel with time format change
import utils.calculators as calculators
import factor_gen.factor101


# In[3]:


def load_data(cols,trade_time):
    file = os.path.join(root,'raw_data','Px_new.mat')
    data1 = mat73.loadmat(file)  
    data1 = data1['Px']

    col = data1['LocalID']
    
    ind = tools.ConTimeDay(data1) 
    ind = tools.Str2times(ind)
   
    Close = data1['Close']
    Close = pd.DataFrame(Close)
    Close.columns  = col
    Close.index = ind

    AMktCap = data1['AMktCap']
    AMktCap = pd.DataFrame(AMktCap)
    AMktCap.columns  = col
    AMktCap.index = ind
    del file,data1
    
    col = cols
    
    
    Close = Close[col]
    AMktCap = AMktCap[col]
    
    Close = Close.iloc[np.where(Close.index<trade_time)]
    AMktCap = AMktCap.iloc[np.where(AMktCap.index<trade_time)]

    names = list(col)

    prices = []
    Close = Close.iloc[-500:]
    for name in names:
        price = Close[name].fillna(0)
        price = price.to_list()
        prices.append(price)

    caps = AMktCap[names]
    caps = AMktCap.iloc[-1]
    caps = caps.fillna(0)
    caps = caps.to_list()
    return names,prices,caps


# In[4]:


# Calculates portfolio mean return
def port_mean(W, R):
    return sum(R * W)

# Calculates portfolio variance of returns
def port_var(W, C):
    return dot(dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)


# In[5]:


# Given risk-free rate, assets returns and covariances, this function calculates
# mean-variance frontier and returns its [x,y] points in two arrays
def solve_frontier(R, C, rf):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(
            mean - r)  # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        return var + penalty

    frontier_mean, frontier_var, frontier_weights = [], [], []
    n = len(R)  # Number of assets in the portfolio
    for r in linspace(min(R), max(R), num=20):  # Iterate through the range of returns on Y axis
        W = ones([n]) / n  # start optimization with equal weights
        b_ = [(0, 1) for i in range(n)]
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        # add point to the efficient frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)
        frontier_var.append(port_var(optimized.x, C))
        frontier_weights.append(optimized.x)
    return array(frontier_mean), array(frontier_var), frontier_weights


# In[6]:


# Given risk-free rate, assets returns and covariances, this function calculates
# weights of tangency portfolio with respect to sharpe ratio maximization
def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)  # calculate mean/variance of the portfolio
        util = (mean - rf) / sqrt(var)  # utility = Sharpe ratio
        return 1 / util  # maximize the utility, minimize its inverse value
    n = len(R)
    W = ones([n]) / n  # start optimization with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights for boundaries between 0%..100%. No leverage, no shorting
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Sum of weights must be 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x


# In[7]:


class Result:
    def __init__(self, W, tan_mean, tan_var, front_mean, front_var, front_weights):
        self.W=W
        self.tan_mean=tan_mean
        self.tan_var=tan_var
        self.front_mean=front_mean
        self.front_var=front_var
        self.front_weights=front_weights
        
def optimize_frontier(R, C, rf):
    W = solve_weights(R, C, rf)
    tan_mean, tan_var = port_mean_var(W, R, C)  # calculate tangency portfolio
    front_mean, front_var, front_weights = solve_frontier(R, C, rf)  # calculate efficient frontier
    # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
    return Result(W, tan_mean, tan_var, front_mean, front_var, front_weights)

def display_assets(names, R, C, color='black'):
    n = len(names)
    scatter([C[i, i] ** .5 for i in range(n)], R, marker='x', color=color), grid(True)  # draw assets
    for i in range(n): 
        text(C[i, i] ** .5, R[i], '  %s' % names[i], verticalalignment='center', color=color) # draw labels

def display_frontier(result: Result, names, label=None, color='black'):
    from collections import defaultdict
    from IPython.core.display import HTML
    text(result.tan_var ** .5, result.tan_mean, '   tangent', verticalalignment='center', color=color)
    scatter(result.tan_var ** .5, result.tan_mean, marker='o', color=color), grid(True)
    plot(list(result.front_var ** .5), list(result.front_mean), label=label, color=color), grid(True)  # draw efficient frontier
    
    table = defaultdict(list)
    for mean, var, weights in zip(result.front_mean, result.front_var, result.front_weights):
        table['Mean'].append(mean)
        table['Variance'].append(var)
        for name, weight in zip(names, weights):
            table[name].append(weight)
    display(HTML(f'<b>Efficient frontier portfolios ({label})</b>'), DataFrame(table))


# In[8]:


# Function takes historical stock prices together with market capitalizations and
# calculates weights, historical returns and historical covariances
def assets_historical_returns_and_covariances(prices):
    prices = np.array(prices)  # create numpy matrix from prices
    # create matrix of historical returns
    rows, cols = prices.shape
    returns = empty([rows, cols - 1])
    for r in range(rows):
        for c in range(cols - 1):
            p0, p1 = prices[r, c], prices[r, c + 1]
            if abs(p0-0)>0 and abs(p1-0)>=0.0000001:
                returns[r, c] = (p1 / p0) - 1
            else:
                returns[r, c] = 0
    
    # calculate returns
    expreturns = array([])
    for r in range(rows):
        expreturns = append(expreturns, np.mean(returns[r]))
    # calculate covariances
    covars = cov(returns)
    expreturns = (1 + expreturns) ** 250 - 1  # Annualize returns
    covars = covars * 250  # Annualize covariances
    return expreturns, covars


# In[9]:


def get_weight(cols,trade_time):
    names, prices, caps = load_data(cols,trade_time)
    n = len(names)
    
    W = np.array(caps) / sum(caps) # calculate market weights from capitalizations
    R, C = assets_historical_returns_and_covariances(prices)
    rf = .0202  # Risk-free rate-一年期债券利率
    
    zero_index = np.array(np.where(W==0))
    not_zero_index= np.array(np.where(W!=0))
    W = W[not_zero_index][0]
    R = R[not_zero_index][0]
    C = np.delete(C,zero_index,axis = 0)
    C = np.delete(C,zero_index,axis = 1)
    names = np.array(names)[not_zero_index][0]
    
    res1 = optimize_frontier(R, C, rf)
    
    #draw
    display_assets(names, R, C, color='blue')
    display_frontier(res1,names, color='blue')
    xlabel('variance $\sigma$'), ylabel('mean $\mu$'), show()
    
    #get result
    df = pandas.DataFrame({'trade_date':trade_time,'sec_code':names,'weight': res1.W})
    
    return df


# In[12]:


if __name__=='__main__':
    cols = ['000001','000002','000004','000005','000006']
    trade_time = '2022-04-03'
    df = get_weight(cols,trade_time)
    print(df)


# In[ ]:




