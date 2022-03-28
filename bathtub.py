import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def exp_dist(t,lambd,beta):
    return (lambd*math.e)**(-lambd*t)+beta
#t=np.linspace(0,8,100)
#plt.plot(t,exp_dist(t,2.0,0.5))
#plt.title('指数分布(λ=1.0)')
for lambd in list(range(2,6))+list(range(10,30,10)):
    lambd/=10
    t=np.linspace(0,8,100)
    plt.plot(t,exp_dist(t,lambd,0.5),label='λ={0:.1f}'.format(lambd))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.savefig('lamdb_arange.png')
plt.show()