import torch
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path='./w2v.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    # 读取之前训练好的 word2vector 模型
    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    # 把 word 加进embedding，并赋予他一个随机生成的表示向量
    def add_embedding(self, word):
        # word 只会是 <PAD> 或 <UNK>
        vector = torch.empty(1, self.embedding_dim) # 生成 1 * dim 维的张量
        torch.nn.init.uniform_(vector) # 从均匀分布中得出的值(0, 1)填充输入张量 vector
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        # 读取之前训练好的 word2vector 模型
        print('Getting embedding ...')
        if load:
            print('loading word to vec model')
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的字典，idx2word 的列表，word2vector 的列表
        for i, word in enumerate(self.embedding.wv.vocab): # model.wv.vocab 获取训练好后所有的词
            print('get words #{}'.format(i + 1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        print('total word: {}'.format(len(self.embedding_matrix)))
        return self.embedding_matrix

    # 将每个句子变成一样的长度
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx['<PAD>'])
        assert len(sentence) == self.sen_len # 断言
        return sentence
    
    # 把句子里的字转成对应的 index
    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
            # 将句子长度变为一致
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    
    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)
            