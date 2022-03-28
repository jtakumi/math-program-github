import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy import special
import japanize_matplotlib 

def exp_dist(t,lambd,beta):
    return (lambd*math.e)**(-lambd*t)+beta

def exp_dist_sym(t,t2,alpha,beta):
    return alpha*(t-t2)+beta

def bathtub_curve(t,lambd,alpha,beta):
    t2=4.0
    if t < 2.0:
        return exp_dist(t,lambd,beta)
    elif t < t2:
        return beta
    else:
        return exp_dist_sym(t,t2,alpha,beta)


t=np.linspace(0,5,100)
plt.plot(t,special.gamma(t))
plt.savefig('bathtub.png')
plt.show()
"""plt.plot(t,exp_dist(t,2.0,0.5))
plt.title('指数分布(λ=1.0)')
for lambd in list(range(2,6))+list(range(10,30,10)):
    lambd/=10
    t=np.linspace(0,8,100)
    plt.plot(t,exp_dist(t,lambd,0.5),label='λ={0:.1f}'.format(lambd))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.savefig('lamdb_arange.png')

t=np.linspace(0,8,100)
plt.plot(t,[bathtub_curve(_t,1.0,0.5,0.5)for _t in t])
plt.savefig('bathtub_a.png')
"""
