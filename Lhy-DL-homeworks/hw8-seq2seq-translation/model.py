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
    def __init__(self)