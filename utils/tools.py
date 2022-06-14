import pandas as pd
import numpy as np
from datetime import datetime
import time



################################### simple dealing
# convert list to matrix， but every column have same value to the list
def Repmat(f_,lowBan):
    z = f_.copy()
    for v in z.columns.tolist():
        z[v] = lowBan
    return z

# Convert a list full of string to python datetime format
def Str2times(indx):
    z = list()
    for v in indx:
        z.append(datetime.strptime(v,'%Y-%m-%d'))
    return z 

def Str2times2(indx):
    z = list()
    for v in indx:
        z.append(datetime.strptime(v,'%Y-%m-%d %H:%M:%S'))
    return z 
    
# Convert matlab date format to python date format.
def ConTimeDay(data1):
    date = data1['Date']
    out = []
    for v in date:
        date=(v-719529)*86400+3600*8
        date = time.localtime(date)
        date = time.strftime('%Y-%m-%d',date)
        out.append(date)
    return out

def ConTimeDay2(date):
    out = []
    for v in date:
        date=(v-719529)*86400+3600*8
        date = time.localtime(date)
        date = time.strftime('%Y-%m-%d',date)
        out.append(date)
    return out

def ConTimeMin(date):
    out = []
    for v in date:
        date=(v-719529)*86400+3600*8
        date = time.localtime(date)
        date = time.strftime('%Y-%m-%d %H:%M:%S',date)
        out.append(date)
    return out




