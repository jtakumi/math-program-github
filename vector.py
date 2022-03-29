import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

score=dict()
score['A']=[55,80]
score['B']=[83,58]
score['C']=[70,95]
score['D']=[50,40]
fig,ax=plt.subplots()
plt.xlabel=('国語(点)')
plt.ylabel=('数学(点)')
plt.xlim([0,100])
plt.ylim([0,100])
for k,v in score.items():
    ax.annotate(k,xy=(v[0],v[1]))
    ax.quiver(0,0,v[0],v[1],angles='xy',scale_units='xy',scale=1)
plt.savefig('vector1.png')
plt.show()