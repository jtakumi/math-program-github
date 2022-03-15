import math

import numpy as np 
import pandas as pd 
from matplotlib import pylab as plt

import japanize_matplotlib

index=list()
data={'平均値':list()}

for i in range(8):
    n=10**(i+1)
    index.append(n)
    data['平均値'].append(np.random.choice(range(1,7),size=n).mean())

print(pd.DataFrame(data,index=index))