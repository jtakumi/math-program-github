import math

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

#matpltlibの日本語化設定
import japanize_matplotlib

#100の座標を作成
t=np.linspace(-np.pi, np.pi,100)
#x軸とy軸の長さが等しくなるように設定
plt.figure(figsize=(5,5))
#x軸はcos,y軸はsinで描画する
plt.plot(np.cos(t),np.sin(t))
plt.show()