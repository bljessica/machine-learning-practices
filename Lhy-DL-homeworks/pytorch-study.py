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

# 张量转换
print('to and from numpy and torch')
print(torch.from_numpy(x_numpy), x_torch.numpy())
print()

# 基本运算
y_numpy = np.array([3, 4, 5])
y_torch = torch.tensor([3, 4, 5])
print('x + y: ', x_numpy + y_numpy, x_torch + y_torch)
print()

# 二范数
print('norm: ')  #
print(np.linalg.norm(x_numpy), torch.norm(x_torch))
print()

# 某维度上进行操作
print('mean along the 0th dimension')
x_numpy = np.array([[1, 3], [3, 4.]])
x_torch = torch.tensor([[1, 3], [3, 4.]])
print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))
print()

# view()
print('view: ')
N ,C, W, H = 10000, 3, 28, 28
X = torch.randn(N, C, W, H)  # 生成 N*C*W*H 的随机数矩阵

print(X.shape)
print(X.view(N, C, 784).shape)
print(X.view(-1, C, 784).shape)  # 自动算 N' * C * 784
print()

# 广播语义
print('broadcasting semantics:')
x = torch.empty(5, 1, 4, 1)
y = torch.empty(   3, 1, 1)
print(x.size(), y.size(), (x + y).size())
print()

# 计算图
print('computation graph:')
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
c = a + b
d = b + 1
e = c * d
print('c', c)
print('d', d)
print('e', e)

# 切换计算设备
# cpu = torch.device('cpu')
# gpu = torch.device('cuda')
# x = torch.rand(10)
# print(x)
# x = x.to(gpu)
# print(x)
# x = x.to(cpu)
# print(x)
# print()

# 计算梯度
print('compute gradients: ')

def f(x):
    return (x - 2) ** 2

def fp(x):
    return 2 * (x - 2)

x = torch.tensor([1.0], requires_grad=True)
y = f(x)
y.backward()  # 自动计算梯度
print('analytical: ', fp(x))
print('pytorch: ', x.grad)
print()

def g(w):
    return 2 * w[0] * w[1] + w[1] * torch.cos(w[0])

def grad_g(w):
    return torch.tensor([2 * w[1] - w[1] * torch.sin(w[0]),
                         2 * w[0] + torch.cos(w[0])])

w  =torch.tensor([np.pi, 1], requires_grad=True)
z = g(w)
z.backward()
print('Analytical:', grad_g(w))
print('Pytorch: ', w.grad)

# 梯度下降
print('gradient descent: ')
x = torch.tensor([5.0], requires_grad=True)
step_size = 0.25
print('iter, \tx, \tf(x), \tf\'(x), \tf\'(x) pytorch')
for i in range(15):
    y = f(x)
    y.backward()

    print('{}, \t{:.3f}, \t{:.3f}, \t{:.3f}, \t{:.3f}'.format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))
    x.data = x.data - step_size * x.grad

    x.grad.detach_()  # 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
    x.grad.zero_()  # 梯度清零
print()

# 线性回归
print('Linear regression: ')
d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + torch.randn(n, 1) * 0.1  # 最后两个维度的数据相乘
print('X shape: ', X.shape, 'y shape: ', y.shape, 'w shape: ', y.shape)
print()

# 更复杂的梯度计算
def model(X, w):
    return X @ w

# 残差平方和
def rss(y, y_hat):
    return torch.norm(y - y_hat) ** 2 / n

def grad_rss(X, y, w):
    return -2 * X.t() @ (y - X @ w) / n

w = torch.tensor([[1.], [0]], requires_grad=True)
y_hat = model(X, w)

loss = rss(y, y_hat)
loss.backward()

print('Analytical gradient: ', grad_rss(X, y, w).detach().view(2).numpy())
print('Pytorch gradient: ', w.grad.view(2).numpy())
print()

# 通过梯度下降来进行线性回归
print('Linear regression using GD: ')
step_size = 0.1

print('iter, \tloss, \tw')
for i in range(20):
    y_hat = model(X, w)
    loss = rss(y, y_hat)

    loss.backward()

    w.data = w.data - step_size * w.grad

    print('{}, \t{:.2f}, \t{}'.format(i, loss.item(), w.view(2).detach().numpy()))

    w.grad.detach()
    w.grad.zero_()

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', w.view(2).detach().numpy())
print()

# 线性模型
print('Linear module:')
d_in = 3
d_out = 4
linear_module = nn.Linear(d_in, d_out)

example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
# 对数据做线性变换
transformed = linear_module(example_tensor)
print('example_tensor: ', example_tensor.shape)
print('transformed: ', transformed.shape)
print('W: ', linear_module.weight)
print('b: ', linear_module.bias)
print()

# 激活函数
print('activation functions: ')
activation_fn = nn.ReLU()
example_tensor = torch.tensor([-1.0, 1.0, 0.0])
activated = activation_fn(example_tensor)
print('examply_tensor', example_tensor)
print('activated', activated)
print()

# 序列化
print('sequential: ')
d_in = 3
d_hidden = 4
d_out = 1
model = torch.nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(),
                                      nn.Linear(d_hidden, d_out),
                                      nn.Sigmoid())
example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
transformed = model(example_tensor)
print('transformed', transformed.shape)

print('parameters: ')
params = model.parameters()

for param in params:
    print(param)
print()

# 损失函数
print('loss function: ')
mse_loss_fn = nn.MSELoss()

input = torch.tensor([[0., 0, 0]])
target = torch.tensor([[1., 0, -1]])

loss = mse_loss_fn(input, target)
print(loss)
print()

# 优化方法
print('optimization method:')
model = nn.Linear(1, 1)

X_simple = torch.tensor([[1.]])
y_simple = torch.tensor([[2.]])

optim = torch.optim.SGD(model.parameters(), lr=1e-2)
mse_loss_fn = nn.MSELoss()

y_hat = model(X_simple)
print('model params before: ', model.weight)
loss = mse_loss_fn(y_hat, y_simple)
optim.zero_grad()  # = detach_() + zero_()
loss.backward()
optim.step()
print('model params after: ', model.weight)
print()

# 线性回归的梯度下降自动迭代
print('Linear regression using GD with automatically computed derivatives: ')
step_size = 0.01

linear_module = nn.Linear(d, 1, bias=False)

loss_func = nn.MSELoss()

optim = torch.optim.SGD(linear_module.parameters(), lr=step_size)

print('iter, \tloss, \tw')

for i in range(20):
    y_hat = linear_module(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward();
    optim.step()

    print('{},\t{:.2f},\t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())
print()

# 随机梯度下降
print('SGD: ')
step_size = 0.01

linear_module = nn.Linear(d, 1)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(linear_module.parameters(), lr=step_size)

print('iter,\tloss,\tw')
for i in range(200):
    rand_idx = np.random.choice(n)
    x = X[rand_idx]

    y_hat = linear_module(x)
    loss = loss_func(y_hat, y[rand_idx])
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 20 == 0:
        print('{},\t{:.2f},\t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())
print()

# 神经网络
# print('neural network: ')
# d = 1
# n = 200
# X = torch.rand(n, d)
# y = 4 * torch.sin(np.pi * X) * torch.cos(6 * np.pi * X ** 2)
# plt.scatter(X.numpy(), y.numpy())  # 绘制散点图
# plt.title('plot of $f(x)$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
#
# plt.show()
#
# # 两个隐层的神经网络，激活函数为Tanh
# step_size = 0.05
# n_epochs = 6000
# n_hidden_1 = 32
# n_hidden_2 = 32
# d_out = 1
#
# neural_network = nn.Sequential(nn.Linear(d, n_hidden_1), nn.Tanh(),
#                                nn.Linear(n_hidden_1, n_hidden_2), nn.Tanh(),
#                                nn.Linear(n_hidden_2, d_out))
# loss_func = nn.MSELoss()
# optim = torch.optim.SGD(neural_network.parameters(), lr=step_size)
# print('iter,\tloss')
# for i in range(n_epochs):
#     y_hat = neural_network(X)
#     loss = loss_func(y_hat, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#
#     if i % (n_epochs // 10) == 0:
#         print('{},\t{:.2f}'.format(i, loss.item()))
# print()
#
# X_grid = torch.from_numpy(np.linspace(0, 1, 50)).float().view(-1, d)
# y_hat = neural_network(X_grid)
# plt.scatter(X.numpy(), y.numpy())
# plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), 'r')
# plt.title('plot of $f(x)$ and $\hat{f}(x)$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()

# 动量
# print('Momentum: ')
# step_size = 0.05
# momentum = 0.9
# n_epochs = 1500
# n_hidden_1 = 32
# n_hidden_2 = 32
# d_out = 1
#
# neural_network = nn.Sequential(nn.Linear(d, n_hidden_1), nn.Tanh(),
#                                nn.Linear(n_hidden_1, n_hidden_2), nn.Tanh(),
#                                nn.Linear(n_hidden_2, d_out))
# loss_func = nn.MSELoss()
# optim = torch.optim.SGD(neural_network.parameters(), lr=step_size, momentum=momentum)
# print('iter,\tloss')
# for i in range(n_epochs):
#     y_hat = neural_network(X)
#     loss = loss_func(y_hat, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#
#     if i % (n_epochs // 10) == 0:  # //向下取整
#         print('{},\t{:.2f}'.format(i, loss.item()))

# 交叉熵损失
print('CrossEntropy loss: ')
loss = nn.CrossEntropyLoss()
input = torch.tensor([[-1., 1], [-1, 1], [1, -1]])

target = torch.tensor([1, 1, 0])
output = loss(input, target)
print(output)
print()

# 卷积
# print('Convolution')
# # an entire mnist digit
# image = np.array([0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3803922 , 0.37647063, 0.3019608 ,0.46274513, 0.2392157 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3529412 , 0.5411765 , 0.9215687 ,0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 ,0.9843138 , 0.9843138 , 0.9725491 , 0.9960785 , 0.9607844 ,0.9215687 , 0.74509805, 0.08235294, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.54901963,0.9843138 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.7411765 , 0.09019608, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8862746 , 0.9960785 , 0.81568635,0.7803922 , 0.7803922 , 0.7803922 , 0.7803922 , 0.54509807,0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 ,0.5019608 , 0.8705883 , 0.9960785 , 0.9960785 , 0.7411765 ,0.08235294, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.14901961, 0.32156864, 0.0509804 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.13333334,0.8352942 , 0.9960785 , 0.9960785 , 0.45098042, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.32941177, 0.9960785 ,0.9960785 , 0.9176471 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.32941177, 0.9960785 , 0.9960785 , 0.9176471 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.4156863 , 0.6156863 ,0.9960785 , 0.9960785 , 0.95294124, 0.20000002, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.09803922, 0.45882356, 0.8941177 , 0.8941177 ,0.8941177 , 0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.94117653, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.26666668, 0.4666667 , 0.86274517,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.5568628 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.14509805, 0.73333335,0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 , 0.8745099 ,0.8078432 , 0.8078432 , 0.29411766, 0.26666668, 0.8431373 ,0.9960785 , 0.9960785 , 0.45882356, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.4431373 , 0.8588236 , 0.9960785 , 0.9490197 , 0.89019614,0.45098042, 0.34901962, 0.12156864, 0., 0.,0., 0., 0.7843138 , 0.9960785 , 0.9450981 ,0.16078432, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.6627451 , 0.9960785 ,0.6901961 , 0.24313727, 0., 0., 0.,0., 0., 0., 0., 0.18823531,0.9058824 , 0.9960785 , 0.9176471 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.07058824, 0.48627454, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.32941177, 0.9960785 , 0.9960785 ,0.6509804 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.54509807, 0.9960785 , 0.9333334 , 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8235295 , 0.9803922 , 0.9960785 ,0.65882355, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.9490197 , 0.9960785 , 0.93725497, 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.34901962, 0.9843138 , 0.9450981 ,0.3372549 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.01960784,0.8078432 , 0.96470594, 0.6156863 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.01568628, 0.45882356, 0.27058825,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.],
#                  dtype=np.float32)
# image_torch = torch.from_numpy(image).view(1, 1, 28, 28)
#
# gaussian_kernel = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
#
# conv = nn.Conv2d(1, 1, 3)
#
# conv.weight.data[:] = gaussian_kernel
#
# convolved = conv(image_torch)
#
# plt.title('origin image: ')
# plt.imshow(image_torch.view(28, 28).detach().numpy())
# plt.show()
#
# plt.title('blurred image: ')
# plt.imshow(convolved.view(26, 26).detach().numpy())
# plt.show()


# Conv2d() + 通道
print('Convolution with channels')
im_channels = 3
out_channels = 16
kernel_size = 3
batch_size = 4
image_width = 32
image_height = 32

im = torch.randn(batch_size, im_channels, image_width, image_height)
m = nn.Conv2d(im_channels, out_channels, kernel_size)
convolved = m(im)

print('im shape', im.shape)
print('convolved im shape', convolved.shape)

# 数据集
# print('Dataset: ')
# from torch.utils.data import Dataset, DataLoader
#
# class FakeDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
# x = np.random.rand(100, 10)
# y = np.random.rand(100)
#
# dataset = FakeDataset(x, y)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#
# if __name__ == '__main__':
#     for i_batch, samplt_batched in enumerate(dataloader):
#         print(i_batch, samplt_batched)


# 混合精度训练
# from apex import amp
#
# model = torch.nn.Linear(10, 10).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#
# model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
#
# ...
#
# with amp.scale_loss(loss, optimizer) as scaled_loss:
#     scaled_loss.backward()