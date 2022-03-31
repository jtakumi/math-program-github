import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

x=[5,1]
y=[2,3]
naiseki=0
for i in range(len(x)):
    naiseki+=x[i]*y[i]
x2=np.array([5,1])
y2=np.array([2,3])
ns=x2.dot(y2)/(np.linalg.norm(x2)*np.linalg.norm(y2))
print('x*y',naiseki)
print('cos ',ns)