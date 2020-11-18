# 线性回归就是最简单的只有一个神经元的神经网络
import torch

# 1.准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]]) # 3*1的矩阵
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 2.设计模型
class LinearModel(torch.nn.Module): #由Model构造出的对象会自动实现反向传播过程，nn->Neural Network
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #(in_features_size, out_features_size)(Linear也继承自Model)

    # 重载
    def forward(self, x):  # 前馈
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# 3.构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4.训练循环
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)  #前馈
    print(epoch, loss)

    optimizer.zero_grad()  #梯度清零
    loss.backward()  #反向传播
    optimizer.step()  #更新

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)