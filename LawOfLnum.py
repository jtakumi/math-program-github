
import math

import numpy as np 
import pandas as pd 
from matplotlib import pylab as plt

import japanize_matplotlib

#平均値
means=list()
#サイコロを振る
numbers=np.array(list())

for n in np.random.choice(range(1,7),size=500):
    numbers=np.append(numbers,n)
    means.append(numbers.mean())

plt.title('サイコロを５００回振った時の平均値')
plt.plot(means)
plt.axhline(y=3.5,color='r',linestyle='--')
plt.show()