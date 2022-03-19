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

rads=[0.0,0.5,1.0,1.5]
df=pd.DataFrame([[sin(r),np.sin(r)] for r in rads],index=rads,columns=['自作関数','numpy'])
df.to_csv('../math-program-github/sin.csv')
print(df)