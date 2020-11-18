from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.5, 4.5, 6.5]

def forward(x, b):
    return x * w + b

def loss(x, y, b):
    y_pred = forward(x, b)
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1): #arange创建等差数组（start, end, step），不含4.1
    for b in np.arange(-1.0, 1.1, 0.1):
        print('w=', w, 'b=', b)
        l_sum = 0
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val, b)
            loss_val = loss(x_val, y_val, b)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE = ', l_sum / 3) #均方误差
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)

#画图
# plt.plot(w_list, mse_list)
# plt.xlabel('w')
# plt.ylabel('Loss')
# plt.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(w_list, b_list, mse_list)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
plt.show()
