import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy import integrate

import japanize_matplotlib 

def gamma(z,t):
    return (t**(z-1))*(math.e ** -t)
#独自実装ガンマ関数
def gamma_rieman_integral(z):
    s=0
    t=np.linspace(0,10000,10000)
    for i in range(len(t)-1):
        s+=gamma(z,t[i])*(t[i+1]-t[i])
    return s

idx=list()
data=list()

for n in range(2,11):
    idx.append(n)
    #factorical関数を使用
    y,abserr=integrate.quad(lambda t:gamma(n,t),0,10000)
    data.append([gamma_rieman_integral(n),math.factorial(n-1),y])

df=pd.DataFrame(data,index=idx,columns=['独自実装ガンマ関数','math.factorical','scipy.integrate.quad'])
df.to_csv('../math-program-github/gamma.csv')
print(df)


