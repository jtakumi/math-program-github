import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def cos(z,n_lim=100):
    _cos=0

    for n in range(n_lim):
        m=2*n
        _cos+=(((-1)**n)/math.factorial(m)) *(z**m)

    return _cos

rads=[0.0,0.5]
df=pd.DataFrame([[cos(r),np.cos(r)] for r in rads],index=rads,columns=['自作関数','numpy'])
df.to_csv('../math-program-github/cos.csv')
print(df)