import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

baseball=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv')
soccer=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/サッカー/Jリーグ選手身長体重.csv')
sumou=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/相撲/力士身長体重.csv')

baseball.plot(kind='scatter',x="身長",y="体重",title="野球選手の身長/体重")
f='covariance_baseball.png'
plt.savefig(f)
soccer.plot(kind='scatter',x="身長",y="体重",title="Jリーグ選手の身長/体重")
f='covariance_soccer.png'
plt.savefig(f)
sumou.plot(kind='scatter',x="身長",y="体重",title="力士の身長/体重")
f='covariance_sumou.png'
plt.savefig(f)
print("画像保存が完了しました。")

def cov(x,y):
    #平均を求める
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    n=len(x)
    c=0.0

    for i in range(n):
        c+=(x[i]-x_mean)*(y[i]-y_mean)
    return c/n

df=pd.DataFrame({'共分散':[
    cov(baseball['身長'],baseball['体重']),
    cov(soccer['身長'],soccer['体重']),
    cov(sumou['身長'],sumou['体重']),
]},index=['野球','サッカー','相撲'])

df.to_csv('../math-program-github/CovarianveCsv.csv')