import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def sin(z,n_lim=100):
    _sin=0

    for n in range(n_lim):
        m=2*n+1
        _sin+=(((-1)**n)/math.factorial(m)) *(z**m)

    return _sin

def cos(z,n_lim=100):
    _cos=0

    for n in range(n_lim):
        m=2*n
        _cos+=(((-1)**n)/math.factorial(m)) *(z**m)

    return _cos


def tan(z,n_lim=100):
    return sin(z,n_lim)/cos(z,n_lim)

rads=[1,1.5]
df=pd.DataFrame([[tan(r),np.tan(r)] for r in rads],index=rads,columns=['自作関数','numpy'])
df.to_csv('../math-program-github/tan.csv')
print(df)