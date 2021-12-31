import math

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

#matpltlibの日本語化設定
import japanize_matplotlib

t=np.linspace(-np.pi,np.pi,100)
#x,y軸の長さが等しくなるように調整
plt.figure(figsize=(5,5))
#x=cos,y=sin
plt.plot(np.cos(t),1/2*np.sin(t))
plt.ylim((-1.0,1.0))
plt.show()