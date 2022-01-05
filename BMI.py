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
print(baseball.head(3))
print(sumou.head(3))
#pandasで標準偏差を計算
baseball_std=np.std(baseball['BMI'])
sumou_std=np.std(sumou['BMI'])
print(pd.DataFrame({'標準偏差 ':[baseball_std,sumou_std]},index=['野球','相撲']))