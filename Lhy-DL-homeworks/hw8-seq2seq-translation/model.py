import torch
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


# 当输出过长或单独靠内容向量无法取得整个输入的意思时，用 Attention 机制来提供 Decoder 更多信息
# 主要是根据现在 Decoder hidden state 去计算 Encoder outputs 中那些与其有较大关系，根据关系的数值来决定该传给 Decoder 的额外信息
class Attention(mm.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
    
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch_size, seq_len, hid_dim * directions]
        # decoder_hidden = [num_layers, batch_size, hid_dim]
        # 一般是取 encoder 最后一层的 hidden state 来做 Attention
        # TODO：实际Attention机制
        attention = None
        return attention


# 由 encoder 和 decoder 组成
class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch_size, input_len, vocab_size]
        # target = [batch_size, target_len, vocab_size]
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input)
        # encoder 的最终隐层 hidden state 用来初始化 decoder
        hidden = hidden.view(self.encoder.n_layers, 2, )