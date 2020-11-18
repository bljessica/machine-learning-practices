import torch
import torch.nn.functional as F

# in_channels, out_channels = 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1

# pytorch中的输入数据必须是小批量的数据(batch_size)
# input = torch.randn(batch_size, in_channels, width, height)  # 从正态分布进行采样生成随机数
input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)

# 创建卷积对象
# conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)  # 填充一圈0
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)  # 每次移两格
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data

# MaxPooling Layer最大池化层（取每个区域内的最大值）
# input = [3, 4, 6, 5,
#          2, 4, 6, 8,
#          1, 6, 7, 8,
#          9, 7, 4, 6]
# input = torch.Tensor(input).view(1, 1, 4, 4)
#
# maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = conv_layer(input)

# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)
print(output)

# 重构MNIST模型 mnist-cnn