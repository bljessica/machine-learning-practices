import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = dropout

    def forward(self, input):
        embedding = self.embedding
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch_size, sequence_len, hid_dim * directions]
        # hidden = [num_layers * drirections, batch_size, hid_dim]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
        self.isatt = isatt # 是否使用 Attention Mechanism
        self.attention = Attention(hid_dim)

        # 使用 Attention 机制会使得输入维度变化，因此输入维度改为：
        self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = dropout
    
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size, vocab_size]
        # hidden = [batch_size, n_layers * directions, hid_dim]
        # decoder 只会是单向的，所以 directions = 1
        input = input.unsqeeze(1) # 在第一维增加一个维度
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch_size, 1, emb_dim]
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch_size, 1, hid_dim]
        # hidden = [num_layers, batch_size, hid_dim]

        # 将 RNN 的输出转成每个词出现的概率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch_size, vocab_size]
        return prediction, hidden


