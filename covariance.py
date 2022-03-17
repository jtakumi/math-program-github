import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

baseball=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/プロ野球/プロ野球選手身長体重.csv')
soccer=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/サッカー/Jリーグ選手身長体重.csv')
sumou=pd.read_csv('C:/Users/Takum/programing/math-program-github/math-program-book-master/9_data/相撲/力士身長体重.csv')

baseball.plot(kind='scatter',x="身長",y="体重",title="野球選手の身長/体重")
f='covariance_baseball.png'
plt.savefig(f)
soccer.plot(kind='scatter',x="身長",y="体重",title="Jリーグ選手の身長/体重")
f='covariance_soccer.png'
plt.savefig(f)
sumou.plot(kind='scatter',x="身長",y="体重",title="力士の身長/体重")
f='covariance_sumou.png'
plt.savefig(f)
