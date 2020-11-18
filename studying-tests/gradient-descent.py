import matplotlib.pyplot as plt
import random

#训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #初始权重猜测

def forward(x):
    return x * w

#梯度下降代价函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(x_data, y_data):
        cost += (forward(x) - y) ** 2
    return cost / len(xs)

#随机梯度下降(SGD)代价函数
def stochastic_cost(x, y):
    return (forward(x) - y) ** 2

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * (forward(x) - y) * x
    return grad / len(xs)

#随机梯度下降梯度
def stochastic_gradient(x, y):
    return 2 * (forward(x) - y) * x

print('Predict(before training)', 4, forward(4))

epoch_list = []
loss_list = []
#进行100轮训练(梯度下降算法)
# for epoch in range(100): #epoch 阶段（轮）
#     cost_val = cost(x_data, y_data)
#     grad_val = gradient(x_data, y_data)
#     w -= 0.01 * grad_val
#     print('Epoch: ', epoch, 'w=', w, 'loss=', cost_val)
#     epoch_list.append(epoch)
#     loss_list.append(cost_val)

#进行100轮训练(随机梯度下降算法)
for epoch in range(100):
    randIndex = random.randint(0, len(x_data) - 1) #随机取一个数据
    x = x_data[randIndex]
    y = y_data[randIndex]
    grad = stochastic_gradient(x, y)
    w -= 0.01 * grad
    print('\tgrad:', x, y, grad)
    loss = stochastic_cost(x, y)
    epoch_list.append(epoch)
    loss_list.append(loss)
    print('Epoch: ', epoch, 'w=', w, 'loss=', loss)
    
print('Predict(after training)', 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()



