import imp
import math
from tkinter.ttk import LabelFrame

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

dataset=pd.DataFrame({'x1':[1.5,2,3,1.5,0.5,-1,-2,-3,-1.5,0],
                    'x2':[1,2.5,3,-2,2,-3,-1.2,-0.5,2,-1.5],
                    'label':['A','A','A','A','A','B','B','B','B','B'],
                    'label_index':[1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0,]})
dataset.to_csv('perceptron_dataset.csv')
#LabelAのプロット
label_a=dataset[dataset['label_index']==1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label A (1)',marker='o')
#LabelBのプロット
label_b=dataset[dataset['label_index']==-1]
plt.scatter(label_a['x1'],label_a['x2'],label='Label B (-1)',marker='x')

#決定境界のプロット
line_x=np.linspace(-4,4,4)
plt.plot(line_x,line_x*-1,'r-')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-3.5,3.5])
plt.ylim([-3.5,3.5])
plt.legend()
plt.savefig('perceptron.png')
plt.show()

print(dataset)