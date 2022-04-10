import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy.spatial.distance import euclidean
import japanize_matplotlib 

baseball=pd.read_csv("C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv")
cov_mtrx=np.cov(baseball['身長'],baseball['体重'])
cov_mtrx_inv=np.array([
    [cov_mtrx[1][1],-cov_mtrx[0][1]],
    [-cov_mtrx[1][0],cov_mtrx[0][0]],
]) / (cov_mtrx[0][0]*cov_mtrx[1][1]- cov_mtrx[0][1]*cov_mtrx[1][0])

def mahal(x,y,cov_mtrx_inv):
    return np.sqrt((x-y).dot(cov_mtrx_inv).dot((x-y).T))

a=np.array([180,110])
b=np.array([200,110])
avg=np.array([np.mean(baseball['身長']),np.mean(baseball['体重'])])
df=pd.DataFrame([[euclidean(avg,a),mahal(avg,a,cov_mtrx_inv)],
            [euclidean(avg,b),mahal(avg,b,cov_mtrx_inv)]],
            index=['選手A','選手B'],columns=['ユーグリッド距離','マハラノビス距離'])
df.to_csv('mahal.csv')
print(df)

baseball_cov=np.cov(baseball['身長'],baseball['体重'])[0][1]
baseball.plot(kind='scatter',x='身長',y='体重',
            title='野球選手の身長/体重の共分散={0}'.format(baseball_cov))
plt.scatter(*a,c='g',marker='^',label='選手A')
plt.scatter(b[0],b[1],c='y',marker='^',label='選手B')
plt.scatter(avg[0],avg[1],c='r',marker='^',label='平均')
plt.legend()
plt.savefig('mahal.png')
plt.show()