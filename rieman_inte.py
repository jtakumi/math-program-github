import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def f(x):
    return 2**x

def one_sixth(alpha,beta):
    area_square=(beta-alpha)*(alpha**2)
    area_a=((beta-alpha)**3)/6
    return area_square-area_a

def rieman_inte(alpha,beta,n):
    d=(beta-alpha)/n
    s=0
    for i in range(n):
        s+=d*f(alpha+(i+1)*d)
    return s

t=np.linspace(0,5,100)
plt.plot(t,f(t),color='b')
plt.fill_between(t,np.zeros(100),f(t),facecolor='b',alpha=0.1)
plt.savefig("rieman_inte.png")

print("one_sixth")
print(one_sixth(-5,5))
print("rieman_inte")
print(rieman_inte(-5,5,100))

plt.show()