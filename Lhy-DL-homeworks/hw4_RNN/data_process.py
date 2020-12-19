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
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 把 word 加进embedding，并赋予他一个随机生成的表示向量
        # word 只会是 <PAD> 或 <UNK>
        vector = torch.empty(1, self.embedding_dim) # 生成 1 * dim 维的张量
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
