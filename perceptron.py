import imp
import math
import copy
from tkinter.ttk import LabelFrame

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 
#sample_dataset
dataset=pd.DataFrame({'x1':[1.5,2,3,1.5,0.5,-1,-2,-3,-1.5,0],
                    'x2':[1,2.5,3,-2,2,-3,-1.2,-0.5,2,-1.5],
                    'label':['A','A','A','A','A','B','B','B','B','B'],
                    'label_index':[1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0]})

#formula_A
def discriminant(p,w):
    return np.dot(p,w)
#formula_B
def activate(x):
    if -1<x:
        return 1
    else:
        return -1

data_size=len(dataset.index)
x=np.array(dataset.loc[:,['x1','x2']])
w=[1,1]

#正解ラベル
label_answer=np.array(dataset['label_index'])
#識別関数の値を初期化
output_test=np.zeros(data_size)
#予測ラベル
label_test=np.zeros(data_size)
#予測結果の正否
result=np.zeros(data_size)

#データセットからラベルを予測する
for i in range(data_size):
    output_test[i]=discriminant(x[i],w)
    label_test[i]=activate(output_test[i])
    result[i]=label_answer[i]==label_test[i]

#output
dataset_test=copy.deepcopy(dataset)
dataset_test['識別関数の値']=output_test
dataset_test['予測ラベル']=label_test
dataset_test['予測の正否']=result
#sample_data_show
print(dataset)

print(dataset_test)


#formula_E
error=0

for i in range(data_size):
    if result[i]==0:
        error-=output_test[i]*label_answer[i]
error_test=pd.DataFrame([[np.sum(result==0),error]],columns=['誤分類の個数','誤差'])
print(error_test)

#csv_output
dataset_test.to_csv('pcp_data_test.csv')
dataset.to_csv('pcp_dataset.csv')
error_test.to_csv('pcp_error.csv')

#learning program
#学習率
eta=0.1
#最大試行回数
max_iter=100

#データセットからラベルを予測する
for i in range(max_iter):
    output_test=np.zeros(data_size)
    label_test=np.zeros(data_size)
    result=np.zeros(data_size)

    #データセットから予測ラベルの出力
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
    print('係数ベクトルの値:w0={0:2f},w1={1:.2f}'.format(w[0],w[1]))
    print()

    #すべて正解ラベルに分類出来たら試行終了
    if miss_classfication==0:
        break

    #係数の更新(確率的勾配降下法)
    for j in np.random.permutation(np.arange(data_size)):
        if result[j]==0.0:
            w+=eta*x[j]*label_answer[j]
            break

#before data
#LabelAのプロット
label_a=dataset[dataset['label_index']==1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label A (1)',marker='o')
#LabelBのプロット
label_b=dataset[dataset['label_index']==-1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label B (-1)',marker='x')

#決定境界のプロット
line_x=np.linspace(-4,4,4)
plt.plot(line_x,line_x*-1,'r-')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-3.5,3.5])
plt.ylim([-3.5,3.5])
#before data
plt.legend()
plt.savefig('perceptron_before.png')
plt.clf()

#After_data
#グラフ描画
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-3.5,3.5])
plt.ylim([-3.5,3.5])
#LabelAプロット
plt.scatter(label_a['x1'],label_a['x2'],label='Label A (1)',marker='o')
#LabelBのプロット
plt.scatter(label_a['x1'],label_a['x2'],label='Label B (-1)',marker='x')
#決定境界のプロット
line_x1=np.linspace(-4,4,4)
plt.plot(line_x1,-1*line_x1*w[0]/w[1],'r-')
plt.legend()
plt.savefig('pcp_after.png')
plt.show()