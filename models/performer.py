import math

import torch
from performer_pytorch import Performer
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        e = Variable(self.pe[:, :x.size(1)],
                     requires_grad=False)
        x = x + e
        return self.dropout(x)


class PerformerSED(nn.Module):
    def __init__(self, dim=768, depth=2, heads=12, out=10):
        super(PerformerSED, self).__init__()
        self.pe = PositionalEncoding(dim)
        self.performer = Performer(dim=dim,
                                   depth=depth,
                                   heads=heads,
                                   dim_head=dim // heads,
                                   causal=True)
        self.linear1 = nn.Linear(16384, out) #65536

    def forward(self, x):
        x = self.pe(x)
        x = self.performer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x
    
    @staticmethod
    def re_init_last_layer(performerSED, out=10):
        pre_last_dim = performerSED.linear1.in_features
        performerSED.linear1 = nn.Linear(pre_last_dim, out)
        torch.nn.init.xavier_uniform_(performerSED.linear1.weight)
        return performerSED

