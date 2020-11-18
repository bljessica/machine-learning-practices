import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#一元线性模型
# w = torch.Tensor([1.0]) #创建Tensor变量
# w.requires_grad = True
#
#
# def forward(x):
#     return x * w # x自动类型转换为Tensor类型
#
# def loss(x, y):
#     return (forward(x) - y) ** 2
#
# print('Predict(before training)', 4, forward(4).item())
#
# w_list = []
# loss_list = []
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l = loss(x, y)
#         l.backward() #反向传播，将链路上所有需要梯度的地方都求出来,存到变量w中
#         print('\tgrad:', x, y, w.grad.item())
#         w.data = w.data - 0.01 * w.grad.data
#         w_list.append(w.data)
#         loss_list.append(l.item())
#
#         w.grad.data.zero_() #将权重里的梯度数据清零
#
#     print('progress:', epoch, l.item())
#
# print('predict(after training)', 4, forward(4).item())
#
# plt.plot(w_list, loss_list)
# plt.xlabel('epoch')
# plt.ylabel('Loss')
# plt.show()

#二次模型
w1 = torch.Tensor([1.0]) #创建Tensor变量
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return w1 * x * x + w2 * x + b # x自动类型转换为Tensor类型

def loss(x, y):
    return (forward(x) - y) ** 2

print('Predict(before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() #反向传播，将链路上所有需要梯度的地方都求出来,存到变量w中
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_() #将权重里的梯度数据清零
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print('progress:', epoch, l.item())

print('predict(after training)', 4, forward(4).item())