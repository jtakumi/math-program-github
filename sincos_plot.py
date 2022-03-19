import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

point_range=5*np.pi
points=np.linspace(-point_range, point_range,100)
plt.plot(np.sin(points))
plt.plot(np.cos(points))
plt.title('sin,cos')

plt.savefig('sincos_plot.png')
plt.show()