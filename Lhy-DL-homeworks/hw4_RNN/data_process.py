from torch import nn
from gensim.models import Word2Vec

class Process():
    def __init__(self, sentences, sen_len, w2v_path='./w2v.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取之前训练好的 word2vector 模型
        self.embedding_matrix = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
        # self.embedding_dim = self.embedding_matrix.vector_size

    def add_embedding(self, word):
        # vector = 
