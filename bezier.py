import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def comb(n,i):
    return math.factorial(n)/(math.factorial(i)*math.factorial(n-i))

def bezier(b,t):
    n=len(b)
    point=0

    for i in range(n):
        point+=b[i]*comb(n-1,i)*((1-t)**(n-i-1)*(t**i))

    return point

points=np.array([[0,0],[1,1],[2,-1],[3,-4]])
x_points=list()
y_points=list()
for t in np.linspace(0,1,100):
    x_points.append(bezier(points[:,0],t))
    y_points.append(bezier(points[:,1],t))

plt.plot(x_points,y_points)
plt.scatter(x=points[:,0],y=points[:,1],color='r')
plt.plot()
plt.savefig('bezier.png')
plt.show()