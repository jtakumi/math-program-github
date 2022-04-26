import imp
import math
import copy
import csv

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib
from pyparsing import sgl_quoted_string 

#野球選手のデータ
baseball=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv")
baseball=baseball.rename(columns={'身長':'x1','体重':'x2'})
baseball['label']='野球'
baseball['label_index']=1.0
baseball['x0']=1
#力士のデータ
sumou=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/相撲/力士身長体重.csv")#[['身長'],['体重']]
sumou=sumou.rename(columns={'身長':'x1','体重':'x2'})
sumou['label']='野球'
sumou['label_index']=1.0
sumou['x0']=1
#学習用データ
train_data=pd.concat([baseball.iloc[0:750,:],
                    sumou.iloc[0:50,:]]).loc[:,['label','label_index','x0','x1','x2']]
#テスト用データ
test_data=pd.concat([baseball.iloc[0:750,:],
                    sumou.iloc[0:50,:]]).loc[:,['label','label_index','x0','x1','x2']]

#シグモイド関数
def std_sigmoid(x):
    return 1/(1+np.exp(-x))
#ステップ関数
def step(x):
    return np.array(x>=0)

#識別関数
def discriminant(p,w):
    return np.dot(p,w)

#予測
def predict(p,w):
    return std_sigmoid(discriminant(p,w))

def log(x):
    return math.log(max(x,1.0E-20))

#formula_B
def activate(x):
    if -1<x:
        return 1
    else:
        return -1

data_size=len(train_data)
x=np.array(train_data.loc[:,['x0','x1','x2']])
w=[1,0,0]
#学習率
eta=0.00001
#予測の出力値
y=np.zeros(data_size)
#正解ラベル
label_answer=np.array(train_data['label_index'])
#最大試行回数
max_iter=30

#識別関数の値を初期化
output_label=np.zeros(data_size)
#予測ラベル
label_test=np.zeros(data_size)
#予測結果の正否
result=np.zeros(data_size)
#コスト
cost_sum=np.zeros(max_iter)

#データセットから予測値を計算
with open('logi_bb_entropy.txt','w',encoding='utf-8') as pe:
    for iter_ in range(max_iter):
        for i in np.random.permutation(np.arange(data_size)):
            #予測値の計算
            output_label[i]=discriminant(x[i],w)
            y[i]=std_sigmoid(output_label[i])
            #誤差の更新(確率的勾配降下法)
            w -= eta * (y[i] - label_answer[i]) * x[i]

        #交差エントロピー誤差
        error=0
        for i in range(data_size):
            error-=label_answer[i] * log(y[i]) + (1-label_answer[i]) * log(1 - y[i])

        cost_sum[iter_] = error / data_size

        print('<{0}回目>'.format(iter_+1),file=pe)
        print('誤差関数の値:{0:2f}'.format(error),file=pe)
        print('交差エントロピー誤差:{0}'.format(cost_sum[iter_]),file=pe)
        print('係数ベクトルの値:w0={0:2f},w1={1:.2f}'.format(w[0],w[1]),file=pe)
        print(file=pe)

#グラフの出力
plt.title('図7:テストデータの予測結果')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([100,250])
plt.ylim([50,250])




#LabelAのプロット
label_a=test_data[test_data['label_index']==1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label A',marker='o')
#LabelBのプロット
label_b=test_data[test_data['label_index']==-0]
plt.scatter(label_a['x1'],label_a['x2'],label='Label B',marker='x')
plt.legend()

#決定境界のプロット
line_x1=np.linspace(50,250,200)
plt.plot(line_x1,-1 * (line_x1 * w[1] + w[0]) / w[2],'r-')
#コスト曲線の出力
plt.figure()
plt.title('図6:平均交差エントロピー誤差')
x_sum=np.arange(max_iter)
plt.plot(x_sum,cost_sum[x_sum],'-r')
#save グラフ
plt.savefig('logi_bb_av_entropy.png')
#グラフの画面出力
plt.show()