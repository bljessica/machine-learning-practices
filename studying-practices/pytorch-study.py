import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

# 设置随机数种子
torch.manual_seed(446)
np.random.seed(446)

# 创建张量
x_numpy = np.array([0.1, 0.2, 0.3])
x_torch = torch.tensor([0.1, 0.2, 0.3])
print('x_numpy', x_numpy)
print('x_torch', x_torch)
print()
