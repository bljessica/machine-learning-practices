# 循环神经网络（处理有序列关系的数据，如天气、股市、自然语言等）
import torch

# hello -> ohlol
input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

# 字符的独热编码
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]


# 用RNNCell
# inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# labels = torch.LongTensor(y_data).view(-1, 1)


# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size):
#         super(Model, self).__init__()
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
#
#     def forward(self, input, hidden):
#         hidden = self.rnncell(input, hidden)
#         return hidden
#
#     # 构造h0时用到
#     def init_hidden(self):
#         return torch.zeros(self.batch_size, self.hidden_size)
#
#
# net = Model(input_size, hidden_size, batch_size)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
#
# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()
#     print('Predict string: ', end='')
#     for input, label in zip(inputs, labels):
#         hidden = net(input, hidden)
#         loss += criterion(hidden, label)  # 要构造计算图
#         _, idx = hidden.max(dim=1)
#         print(idx2char[idx.item()], end='')
#     loss.backward()
#     optimizer.step()
#     print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))


# 用RNN
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))