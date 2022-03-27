import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy import integrate,special

import japanize_matplotlib 

def beta(x,y,t):
    return t**(x-1)*((1-t)**(y-1))

b,abserr=integrate.quad(lambda t:beta(5,2,t),0,1)
x=5
y=2
sg=(special.gamma(x)*special.gamma(y)/special.gamma(x+y))
sb=special.beta(5,2)
print('special.beta',sb)
print('独自実装',b)
print('special.gamma',sg)

