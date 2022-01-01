import math

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

#matpltlibの日本語化設定
import japanize_matplotlib

#野球選手のデータをcsvファイルをpandasライブラリで読み込む
df_b=pd.read_csv('D:\java\math-program-github\math-program-book-master\9_data\プロ野球\プロ野球選手身長体重.csv')
#グラフのデザインと野球選手のデータの描写を決定
ax=df_b[['身長','体重']].plot(kind='scatter',x='身長',y='体重',
                          color='blue',alpha=0.3,
                          title='グラフ1:野球選手と力士の身長体重分布(青=野球選手,赤=力士)')
#力士のデータをcsvファイルで取得
df_s=pd.read_csv('D:\java\math-program-github\math-program-book-master\9_data\相撲\力士身長体重.csv')
#力士のデータ描写を決定
df_s[['身長','体重']].plot(kind='scatter',x='身長',y='体重',
                          color='red',alpha=0.3,ax=ax)
plt.show()
