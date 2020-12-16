import os
from gensim.models import word2vec

from utils import *


# 将词语训练为词向量
def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=5, iter=10, sg=1)
    return model

if __name__ == '__main__':
    print('loading training data...')
    train_x, y = load_training_data()
    train_x_nolabel = load_training_data('training_nolabel.txt')

    print('loading testing data...')
    test_x = load_testing_data()

    model = train_word2vec(train_x + test_x)

    print('saving model...')
    model.save('w2v_all.model')