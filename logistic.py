import imp
import math
import copy
import csv

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib
from pyparsing import sgl_quoted_string 

dataset=pd.DataFrame({'x0':[1,1,1,1,1,1,1,1,1,1],
                      'x1':[1.5,2,3,1.5,0.5,-1,-2,-3,-1.5,0],
                      'x2':[1,2.5,3,-2,2,-3,-1.2,-0.5,2,-1.5],
                      'label':['A','A','A','A','A','B','B','B','B','B'],
                      'label_index':[1.0,1.0,1.0,1.0,1.0,0,0,0,0,0]})

#シグモイド関数
def std_sigmoid(x):
    return 1/(1+np.exp(-x))
#ステップ関数
def step(x):
    return np.array(x>=0)

#識別関数
def discriminant(p,w):
    return np.dot(p,w)

#シグモイド関数のグラフを表示
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.text(-6,0.3,'ラベルBの確率が高い')
plt.text(1.5,0.7,'ラベルAの確率が高い')
plt.title('pic1:ロジスティック回帰(シグモイド関数)のグラフ')
t=np.linspace(-6,6,100)
plt.plot(t,std_sigmoid(t))
plt.xlabel('入力')
plt.ylabel('出力:ラベルAの確率')
plt.plot(0,0.5,'o')
plt.annotate('ラベルAの確率が50%',xy=(0,0.5),xytext=(2,0.5),arrowprops=dict(arrowstyle='->'),fontsize=12)

#ステップ関数の表示
plt.subplot(1,2,2)
t=np.linspace(-6,6,100)
plt.plot(t,step(t))
plt.text(-4,0.3,'ラベルB')
plt.text(2,0.7,'ラベルA')
plt.title('pic2:パーセプトロン(ステップ関数)のグラフ')
plt.xlabel('入力')
plt.ylabel('出力:ラベルAの値')
plt.savefig('logistic_sig_step.png')

#特徴ベクトル
x=np.array(dataset.loc[:,['x0','x1','x2']])
#係数ベクトル
w=[0,1,1]
#データ数
data_size=len(dataset.index)
#識別関数の値
output_label=np.zeros(data_size)
#予測の出力値
y=np.zeros(data_size)
#データセットから予測値を計算
for i in range(data_size):
    output_label[i]=discriminant(x[i],w)
    y[i]=std_sigmoid(output_label[i])

dataset_=copy.deepcopy(dataset)
dataset_['識別関数の値']=output_label
dataset_['予測値y']=y
#結果をファイル保存
with open('logistic.txt','w',encoding='utf-8') as f:
    print(dataset_,file=f)
dataset_.to_csv('logistic.csv',encoding='utf-8')

plt.show()