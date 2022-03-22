import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy.special import expit
import japanize_matplotlib 

def logit(x):
    return np.log(x/(1-x))

def logit2(x):
    return np.log(x)-np.log(1-x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

data=list()
points=[4,8,15,16,23,42]

for a in points:
    data.append(logit(expit(a)))

df=pd.DataFrame(data,index=points,columns=['標準シグモイド関数->ロジット'])
df.to_csv('../math-program-github/logit.csv')
print(logit2(sigmoid(4)))
print(df)
