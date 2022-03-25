import imp
import math

import pandas as pd
import numpy as np
from matplotlib import pylab as plt

import japanize_matplotlib 

def gamma(z,t):
    return (t**(z-1))*(math.e ** -t)

print(gamma(5,2))