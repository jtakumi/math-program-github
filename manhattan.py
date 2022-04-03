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


def manhattan(x,y):
    d=0

    for i in range(len(x)):
        d+=np.abs(x[i]-y[i])
    return d

df=pd.DataFrame([[manhattan(tokyo,sendai)],[manhattan(tokyo,nagoya)]],
            index=['東京-仙台','東京-名古屋'],columns=['マンハッタン距離'])
print(df)