import imp
import math
from matplotlib.transforms import Bbox

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

#標準シグモイド関数
def std_sigmoid(x):
    return 1/(1+np.exp(-x))

#シグモイド関数
def sigmoid(x,a):
    return 1/(1+np.exp(-a*x))


t=np.linspace(-6,6,100)
plt.figure(figsize=(10,5))
#標準シグモイドで計算
plt.plot(t,std_sigmoid(t))
plt.savefig('std_sigmoid.png')

t=np.linspace(-5,5,100)
#1~4のカーブの曲線を描写
for a in range(1,5):
    plt.plot(t,sigmoid(t,a),label='a=%d' % a)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.savefig('sigmoid_5curve.png')
