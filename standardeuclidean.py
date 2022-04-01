import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy.spatial.distance import euclidean
import japanize_matplotlib 

tokyo=[35.709*111,139.732*91*10000]
nagoya=[35.181*111,136.906*91*10000]
sendai=[38.254*111,140.891*91*10000]
std_x=np.std([tokyo[0],nagoya[0],sendai[0]])
std_y=np.std([tokyo[1],nagoya[1],sendai[1]])
std=[std_x,std_y]

def standard_euclidean(x,y,std):
    d=0

    for i in range(len(x)):
        d+=((x[i]-y[i])/std[i])**2
    return np.sqrt(d)

df=pd.DataFrame([[standard_euclidean(tokyo,sendai,std)],[standard_euclidean(tokyo,nagoya,std)]],
            index=['東京-仙台','東京-名古屋'],columns=['標準ユーグリッド距離'])
print(df)