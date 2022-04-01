import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

tokyo=[35.709*111,139.732*91]
nagoya=[35.181*111,136.906*91]
sendai=[38.254*111,140.891*91]

def euclidean(x,y):
    d=0

    for i in range(len(x)):
        d+=(x[i]-y[i])**2
    return np.sqrt(d)

df=pd.DataFrame([[euclidean(tokyo,sendai)],[euclidean(tokyo,nagoya)]],
            index=['東京-仙台','東京-名古屋'],columns=['ユーグリッド距離'])
print(df)