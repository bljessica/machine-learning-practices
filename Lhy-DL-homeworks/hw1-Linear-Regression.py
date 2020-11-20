import sys
import pandas as pd
import numpy as np
# from google.colab import drive

data = pd.read_csv('./train.csv', encoding='big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0