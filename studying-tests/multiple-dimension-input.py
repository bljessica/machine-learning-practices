import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)  # 逗号为分隔符，神经网络计算通常使用32位浮点数
# 从numpy生成Tensor
x_data = torch.from_numpy(xy[:,:-1])  # 所有行，第一列到倒数第二列
y_data = torch.from_numpy(xy[:, [-1]])  # 所有行，最后一列


# 三层神经网络
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入为8维，输出为6维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # self.activate = torch.nn.Sigmoid()  # 激活函数
        self.activate = torch.nn.ReLU()  # 输出为（0，1）,y_hat要再乘一个sigmoid

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    epoch_list.append(epoch)
    loss_list.append(loss.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()