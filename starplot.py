import math

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

#matpltlibの日本語化設定
import japanize_matplotlib

t=np.linspace(-np.pi*5,np.pi*15,500)
plt.figure(figsize=(5,5))
plt.plot(np.cos(t)**3,np.sin(t)**3)
plt.show()