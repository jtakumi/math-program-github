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

data_size=len(dataset.index)
x=np.array(dataset.loc[:,['x0','x1','x2']])
w=[1,0,0]
#学習率
eta=0.05
#誤差
error=0
#予測の出力値
y=np.zeros(data_size)
#正解ラベル
label_answer=np.array(dataset['label_index'])
#最大試行回数
max_iter=40

#識別関数の値を初期化
output_test=np.zeros(data_size)
#予測ラベル
label_test=np.zeros(data_size)
#予測結果の正否
result=np.zeros(data_size)
#コスト
cost_sum=np.zeros(max_iter)
with open('logi_entropy.txt','w',encoding='utf-8') as pe:
    for iter_ in range(max_iter):
        for i in np.random.permutation(np.arange(data_size)):
            #予測値の計算
            y[i]=predict(x[i],w)
            #誤差の更新
            w-=eta*(y[i]-label_answer[i])


        #平均交差エントロピー誤差
        error=0
        for i in range(data_size):
            error-=label_answer[i]*log(y[i])+(1-label_answer[i])*log(1-y[i])

        cost_sum[iter_]=error/data_size

        print('<{0}回目>'.format(iter_+1),file=pe)
        print('誤差関数の値:{0:2f}'.format(error),file=pe)
        print('交差エントロピー誤差:{0}'.format(cost_sum[iter_]),file=pe)
        print('係数ベクトルの値:w0={0:2f},w1={1:.2f}'.format(w[0],w[1]),file=pe)
        print(file=pe)

#グラフの出力
plt.title('図5:サンプルデータと識別境界')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-5,5])
plt.ylim([-5,5])
#LabelAのプロット
label_a=dataset[dataset['label_index']==1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label A (1)',marker='o')
#LabelBのプロット
label_b=dataset[dataset['label_index']==-1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label B (-1)',marker='x')

#決定境界のプロット
line_x1=np.linspace(-4,4,4)
plt.plot(line_x1,-1*(line_x1*w[1]+w[0])/w[2],'r-')
#コスト曲線の出力
plt.figure()
plt.title('図6:平均交差エントロピー誤差')
x_sum=np.arange(max_iter)
plt.plot(x_sum,cost_sum[x_sum],'-r')
plt.xlabel('試行回数')
plt.ylabel('平均誤差')
plt.savefig('logi_av_entropy.png')


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