import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),  # 对图像进行处理，将灰度[0, 255]转成[0, 1]，将维度28*28转成1*28*28的张量
    transforms.Normalize((0.1307, ), (0.3081, ))  # 归一化（均值，标准差）
])

train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ，正确率可达到98%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 池化层
        self.pooing = r=torch.nn.MaxPool2d(2)
        # 全连接层
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # 将数据由(n, 1, 28, 28)展开为(n ,784)
        batch_size = x.size(0)  # 样本数量n
        x = F.relu(self.pooing(self.conv1(x)))
        x = F.relu(self.pooing(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
# 用GPU跑
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)  # 将建立的模型迁移到GPU上

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 带冲量来优化训练过程


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # 用GPU跑
        # inputs, target = inputs.to(device), target.to(device)
        # 初始化
        optimizer.zero_grad()

        # 前馈，反馈，更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试的时候不用计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一个维度（第0个维度是行，第一个维度是列）找最大值，返回值为下标，最大值
            total += labels.size(0)  # N * 1
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()