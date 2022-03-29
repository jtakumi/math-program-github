import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy import integrate

import japanize_matplotlib 

def f(t):
    return np.e **(-t**2)

d=integrate.quad(f,0,1)[0]*(2/np.sqrt(np.pi))
print('error function 独自実装=',d)
erf=0.0
#x=1
for n in range(1000):
    erf+=((-1) ** n * 1 ** (2 * n + 1)) / (math.factorial(n) * (2 * n + 1))
(2 / np.sqrt(np.pi)) * erf
print('function 2=',erf)