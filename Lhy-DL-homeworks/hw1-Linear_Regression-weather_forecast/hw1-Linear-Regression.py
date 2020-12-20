import numpy as np
import pandas as pd

# big5为繁体中文字符标准
data = pd.read_csv('./train.csv', encoding='big5');
data = data.iloc[:, 3:]  # 取第三列及之后的数据
data[data == 'NR'] = 0  # 值为NR则置0
# print(data)

# 转成二维数组
raw_data = data.to_numpy();
# print(raw_data)

# 每个月份的数据（18个指标*（20天*24小时） -> 18*480大小的数组）
month_data = {}
for month in range(12):
    tmp = np.empty([18, 480])
    for day in range(20):
        tmp[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = tmp;
# print(month_data, month_data[0].shape)

# 每九个小时形成一个PM2.5数据
# 只有前九天的PM2.5数据不是根据之前的数据计算的，所以一共有480-9=471组训练数据
# 每九天的18*9个特征就可以得到第十天的PM2.5
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)  # 拉成行向量
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
# print(x, x.shape)
# print(y, y.shape)

# 归一化 -> [0, 1]
mean_x = np.mean(x, axis=0)  #axis = 0：压缩行，对列求均值
std_x = np.std(x, axis=0)
# print(x.shape, mean_x.shape, std_x.shape)
for i in range(len(x)):  # len(x)为行数12*471
    for j in range(len(x[0])):  # 18*9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 划分训练集和验证集
# import math
# x_train_set = x[: math.floor(len(x) * 0.8), :]
# y_train_set = y[: math.floor(len(y)) * 0.8, :]
# x_validation_set = x[math.floor(len(x)) * 0.8:, :]
# y_validation_set = y[math.floor(len(x)) * 0.8:, :]

# 训练
dim = 18 * 9 + 1  # 多一项常数项
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # 拼接数组，维度为列
learing_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 1e-10
for t in range(iter_time):
    # dot()计算乘积
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
    if t % 100 == 0:
        print(str(t) + ':' + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
    adagrad += gradient ** 2
    w = w - learing_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
print(w)

# 测试集
test_data = pd.read_csv('./test.csv', header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
print(test_x)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
print(ans_y)

# 预测结果写入文件
import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)




# 测试



