import torch
import numpy as np
from torch.utils.data import Dataset  # 抽象类
from torch.utils.data import DataLoader


# 糖尿病数据集
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # [行数，列数]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  # 使用索引拿出想要的数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 数据条数
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # shuffle打乱, num_workers读数据的并行进程数


# 神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入为8维，输出为6维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()  # 激活函数
        # self.activate = torch.nn.ReLU()  # 输出为（0，1）,y_hat要再乘一个sigmoid

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':  # 解决windows和linux对于多线程的实现函数不一样的问题
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # data为（x, y）元组
            # 1.准备数据
            inputs, labels = data  # data会被自动转化为Tensor
            # 2.前馈
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3.反向传播
            optimizer.zero_grad()
            loss.backward()
            # 4.更新
            optimizer.step()
