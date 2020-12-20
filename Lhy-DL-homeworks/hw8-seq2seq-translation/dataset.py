import re
import os
import numpy as np
import json
import torch

# 将不同长度的答案扩展为相同长度
class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad
    
    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape(0))), mode='constant', constant_values=self.pad)


class EN2CNDataset(data.Dataset):
    def __init__(self, root, max_output_len, set_name):
        # super().__init__()
        self.root = root
        
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 载入资料
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), 'r') as f:
            for line in f:
                self.data.append(line)
        print(f'{set_name} dataset size: {len(self.data)}')

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        # 拓展长度
        self.transfrom = LabelTransform(max_output_len, self.word2int_en['PAD'])
    
    def get_dictionary(self, language):
        # 载入字典
        with open(os.path.join(self.root, f'word2int_{language}.json'), 'r') as f:
            word2int = json.load(f)
        with open(os.path.join()(self.root, f'int2word_{language}.json'), 'r') as f:
            int2word = json.load(f)
        return word2int, int2word
    
    def __len(self):
        return len(self.data)
    
    def __getitem(self, index):
        # 分开中英文
        sentences = self.data[index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))

        assert len(sentences) == 2

        BOS = self.word2int_en['BOS']
        EOS = self.word2int_en['EOS']
        UNK = self.word2int_en['UNK']

        en, cn = [BOS], [BOS]
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        sentence = re.split(' ', sentence[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)
        # 扩展句子长度
        en, cn = self.transfrom(en), self.transfrom(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn