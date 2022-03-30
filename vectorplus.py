import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 
x=[1,5]
y=[5,3]
answer=[0,0]
for i in range(len(x)):
    answer[i]=x[i]+y[i]
fig, ax=plt.subplots()
plt.xlim([0,10])
plt.ylim([0,10])
ax.annotate('x',xy=(x[0],x[1]))
ax.annotate('y',xy=(y[0],y[1]))
ax.annotate('answer',xy=(answer[0],answer[1]))
ax.quiver(0,0,x[0],x[1],angles='xy',scale_units='xy',scale=1)
ax.quiver(0,0,y[0],y[1],angles='xy',scale_units='xy',scale=1)
ax.quiver(0,0,answer[0],answer[1],angles='xy',scale_units='xy',scale=1)
plt.savefig('vectorplus.png')
plt.show()