import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def softmax(x,j):
    return np.exp(x[j])/np.sum(np.exp(x))

x=np.arange(1,11)
result=list()
x2=np.arange(1,101)
result2=list()

for j in range(10):
    result.append(softmax(x,j))

_, ax1 =plt.subplots()
ax1.plot(result)
ax1.twinx().plot(x,color='r')
plt.savefig('softmax10.png')


for j in range(100):
    result2.append(softmax(x2,j))

_, ax2 =plt.subplots()
ax2.plot(result2)
ax2.twinx().plot(x2,color='r')
plt.savefig('softmax100.png')

plt.show()