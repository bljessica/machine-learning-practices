import torch

# 加载数据
def load_training_data(path='./data/training_label.txt') :
    # 有标签数据
    if 'training_label' in path:
        with open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
            return x, y
    else:
        with open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
            return x

def load_testing_data():
    with open('./data/testing_data.txt', "rt", encoding="utf-8") as f:
        lines = f.readlines()
        X = [''.join(line.strip('\n').split(',')[1:]).strip() for line in lines[1:]]
        X = [line.split(' ') for line in X]
        return X

# 评估
def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    return torch.sum(torch.eq(outputs, labels)).item()