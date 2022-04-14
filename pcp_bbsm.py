import imp
import math
import copy
from tkinter.ttk import LabelFrame

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 
#baseball_dataset
baseball=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv")#[['身長'],['体重']]
sumou=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/相撲/力士身長体重.csv")#[['身長'],['体重']]

data_baseball=baseball.rename(columns={'身長':'x1','体重':'x2'}).assign(label='野球',
                                                                label_index=1.0,x0=1)
data_sumou=sumou.rename(columns={'身長':'x1','体重':'x2'}).assign(label='相撲',
                                                                label_index=-1.0,x0=1)
dataset=pd.concat([data_baseball,data_sumou]).loc[:,['label','label_index','x0','x1','x2']]
#formula_A
def discriminant(p,w):
    return np.dot(p,w)
#formula_B
def activate(x):
    if -1<x:
        return 1
    else:
        return -1

#learning program
#学習率
eta=0.1
#最大試行回数
max_iter=1000

data_size=len(dataset.index)
x=np.array(dataset.loc[:,['x0','x1','x2']])
w=[1,1,1]

#正解ラベル
label_answer=np.array(dataset['label_index'])

#データセットからラベルを予測する
for i in range(max_iter):
    output_test=np.zeros(data_size)
    label_test=np.zeros(data_size)
    result=np.zeros(data_size)
    #データセットから予測ラベルを出力
    for j in range(data_size):
        output_test[j]=discriminant(x[j],w)
        label_test[j]=activate(output_test[j])
        result[j]=label_answer[j]==label_test[j]




    #誤差を求める
    error=0

    for j in range(data_size):
        if result[j]==0.0:
            error-=(output_test[j]*label_answer[j])
    
    #誤分類の個数
    miss_classfication=np.sum(result==0.0)

    print('<{0}回目>'.format(i+1))
    print('誤差関数の値:{0:2f}'.format(error))
    print('誤分類の個数:{0}'.format(miss_classfication))
    print('係数ベクトル:w0={0:2f},w1={1:.2f},w2={2:.2f}'.format(w[0],w[1],w[2]))
    print()

    #すべて正解ラベルに分類出来たら試行終了
    if miss_classfication==0:
        break

    #係数の更新(確率的勾配降下法)
    for j in np.random.permutation(np.arange(data_size)):
        if result[j]==0.0:
            w+=eta*x[j]*label_answer[j]
            break


plt.title('プロ野球選手と力士の身長と体重の分布と決定境界')
plt.xlabel('x1 (身長)')
plt.ylabel('x2 (体重)')
plt.xlim([160,210])
plt.ylim([50,210])
#ラベルAのプロット
plt.scatter(data_baseball['x1'],data_baseball['x2'],label='baseball (1)',
marker='o')
#ラベルBのプロット
plt.scatter(data_sumou['x1'],data_sumou['x2'],label='sumou (-1)',marker='x')
#決定境界のプロット
line_x1=np.linspace(160,210,50)
line_x2=-1*(line_x1*w[1]+w[0])/w[2]
plt.plot(line_x1,line_x2,'r-')

plt.legend()
plt.savefig('pcp_bbsm.png')
plt.show()