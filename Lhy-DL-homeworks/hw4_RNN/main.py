import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from utils import *
from model import LSTM_Net
from data_process import Preprocess
from data import TwitterDataset

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    #  torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
    total = sum(p.numel() for p in model.parameters()) # parameters()返回网络中的参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters: total {}, trainable {}\n'.format(total, trainable))
    # 将 model 的模式转为train, optimizer继续更新参数
    model.train()
    criterion = nn.BCELoss() # 二分类交叉熵损失
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            inputs = inputs.to(device, dtype=torch.long)
            # 因為要传入 criterion，所以要是 float
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 对于tensor变量进行维度压缩，去除维数为1的的维度 
            # 去掉最外面的 dimension，好让 outputs 可以传入 criterion()
            outputs = outputs.squeeze() 
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 计算梯度
            optimizer.step() # 更新模型的参数
            correct = evaluation(outputs, labels) # 计算模型准确率
            total_acc += correct / batch_size
            total_loss += loss.item()
            print('[Epoch {}: {}/{}] loss: {:.3f} acc: {:.3f}'.format(epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\r')
        print('\nTrain | Loss: {:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / batch_size * 100))

        # validation
        model.eval() # 将 model 模式调为eval，固定参数
        # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze() 
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += correct / batch_size
                total_loss += loss.item()

            print('Valid | Loss: {:.5f} Acc: {:.3f}'.format(total_loss / v_batch, total_acc / v_batch * 100))
            # 如果 validation 的结果优于之前所有的結果，就把模型存下來
            if total_acc >= best_acc:
                best_acc = total_acc
                torch.save(model, '{}/ckpt.model'.format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
        print('-----------------------------------------------')
        # 将 model 的模式转为train, optimizer继续更新参数
        model.train() 

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, stype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()
    return ret_output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path_prefix = 'data/'

train_with_label = os.path.join(data_path_prefix, 'training_label.txt')
train_no_label = os.path.join(data_path_prefix, 'training_nolabel.txt')
testing_data = os.path.join(data_path_prefix, 'testing_data.txt')

w2v_path = os.path.join('', 'w2v_all.model')

sen_len = 20 # 句子长度
fix_embedding = True
batch_size = 128
epoch = 5
lr = 0.001
model_dir = ''

# 加载数据
print('loading data...')
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)

# 对 input 和 labels 做预处理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 生成 model 对象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)

# 将数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_x[:180000], train_x[18000:], y[:180000], y[180000:]

# 把 data 生成为数据集供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# data 转为 batch of tensors(shuffle 打乱样本)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# 开始训练
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

# 预测、写入 csv 文件
print('loading testing data ...')
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)

# 写入 csv 文件
tmp = pd.DataFrame({'id': [str(i) for i in range(len(test_x))], 'label': outputs})
print('save csv ...')
tmp.to_csv(os.path.join(data_path_prefix, 'prefict.csv'), index=False)
print('Finish predicting ...')