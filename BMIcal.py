import math

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

#matpltlibの日本語化設定
import japanize_matplotlib

#野球選手のデータをcsvファイルをpandasライブラリで読み込む
baseball=pd.read_csv('D:\java\math-program-github\math-program-book-master\9_data\プロ野球\プロ野球選手身長体重.csv')
#力士のデータをcsvファイルで取得
sumou=pd.read_csv('D:\java\math-program-github\math-program-book-master\9_data\相撲\力士身長体重.csv')
baseball['BMI']=baseball['体重']/((baseball['身長']/100)**2)
sumou['BMI']=sumou['体重']/((sumou['身長']/100)**2)
#標準偏差の計算
def std(x):
    #平均
    mu=np.mean(x)
    #標準偏差
    sigma=0.0
    #要素数
    n=len(x)
    for i in range(n):
        sigma+=(x[i]-mu)**2
    
    sigma/=n
    return np.sqrt(sigma)
#野球選手のデータで計算
baseball_std=std(baseball['BMI'])
#力士のデータで計算
sumou_std=std(sumou['BMI'])
print(print(pd.DataFrame({'標準偏差 ':[baseball_std,sumou_std]},index=['野球','相撲'])))