import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

baseball=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv')
soccer=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/サッカー/Jリーグ選手身長体重.csv')
sumou=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/相撲/力士身長体重.csv')

def cov(x,y):
    #平均を求める
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    n=len(x)
    c=0.0

    for i in range(n):
        c+=(x[i]-x_mean)*(y[i]-y_mean)
    return c/n

def corr(x,y):
    
    return np.cov(x,y)[0][1] / (np.std(x,ddof=1) * np.std(y,ddof=1))

df=pd.DataFrame({'相関係数':[
    corr(baseball['身長'],baseball['体重']),
    corr(soccer['身長'],soccer['体重']),
    corr(sumou['身長'],sumou['体重']),
]},index=['野球','サッカー','相撲'])

df.to_csv('../math-program-github/CorrelationCsv.csv')
print(df)
