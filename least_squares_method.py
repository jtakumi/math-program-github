import imp
import math
import copy

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

baseball=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv")#[['身長'],['体重']]
#プロット描画
baseball.plot(kind='scatter',x='身長',y='体重',title='プロ野球選手の身長と体重の予測モデル')
plt.xlim([160,210])
plt.ylim([50,140])
x_data=np.arange(150,210)

#モデル1の回帰曲線
model1=1.0*x_data-100
plt.plot(x_data,model1,color='orange')
#モデル２の回帰曲線
model2=1.6*x_data-200
plt.plot(x_data,model2,color='green')

def rss(data):
    result=0
    for i in data.iterrows():
        result+=(i[1]['体重']-i[1]['予測値'])**2
    return result

#model1の二乗和誤差
rss_model1=copy.deepcopy(baseball)
rss_model1['予測値']=1.0*rss_model1['身長']-100
#model2の二乗和誤差
rss_model2=copy.deepcopy(baseball)
rss_model2['予測値']=1.6*rss_model1['身長']-200

rss_show=pd.DataFrame([[rss(rss_model1)],[rss(rss_model2)]],index=['モデル1','モデル２'],columns=['二乗和誤差'])
print(rss_show)
plt.savefig('LSM_baseball_error.png')
#係数を求める

def cov(x,y):
    #共分散
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    n=len(x)
    c=0.0

    for i in range(n):
        x_i=x[i]
        y_i=y[i]
        c+=(x_i-x_mean)*(y_i-y_mean)

    return c/n
def std(x):
    #標準偏差
    mu=np.mean(x)
    _std=0.0
    n=len(x)
    #平均と要素の差をとって2乗する
    for i in range(n):
        _std+=(x[i]-mu)**2
    #要素数で割る
    _std=_std/n
    #平方根の計算
    return np.sqrt(_std)

def slope(x,y):
    #係数の計算
    a=cov(x,y)/(std(x)**2)
    return a

def intercept(x,y):
    #切片
    mean_x=np.array(x).mean()
    mean_y=np.array(y).mean()
    a=slope(x,y)
    b=mean_y-mean_x*a
    return b

print("係数=",slope(baseball['身長'],baseball['体重']))
print("切片=",intercept(baseball['身長'],baseball['体重']))
#最小二乗法による係数aと切片bを求める
a=slope(baseball['身長'],baseball['体重'])
b=intercept(baseball['身長'],baseball['体重'])
x_data=np.arange(150,210)
y_data=a*x_data+b
#最小二乗法による回帰曲線の描画
baseball.plot(kind='scatter',x='身長',y='体重',title='最小二乗法によるプロ野球選手の身長と体重の予測モデル')
plt.plot(x_data,y_data,color='red')
plt.savefig('LSM_baseball.png')
plt.show()