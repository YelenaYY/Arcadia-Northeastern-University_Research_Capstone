# Based on PyTorch reference usage: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
# import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len,
                 max_tgt_len, dropout):
        super(TransformerModel, self).__init__()
        # embedding layers
        # self.enc_embedding = nn.Embedding(num_src_vocab, embedding_dim)
        self.enc_embedding = nn.Linear(input_size, embedding_dim)
        self.dec_embedding = nn.Embedding(num_tgt_vocab, embedding_dim)

        # positional encoding layers
        self.enc_pe = PositionalEncoding(embedding_dim, max_len=max_src_len)
        self.dec_pe = PositionalEncoding(embedding_dim, max_len=max_tgt_len)

        # encoder/decoder layers
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # final dense layer
        self.dense = nn.Linear(embedding_dim, num_tgt_vocab)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, src, tgt):
        # src, tgt = self.enc_embedding(src).permute(1, 0, 2), self.dec_embedding(tgt).permute(1, 0, 2)
        src = self.enc_embedding(src)
        # src = src.permute(1, 0, 2)

        tgt = self.dec_embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        src, tgt = self.enc_pe(src), self.dec_pe(tgt)
        memory = self.encoder(src)
        transformer_out = self.decoder(tgt, memory)
        final_out = self.dense(transformer_out)
        # return self.log_softmax(final_out)  # Commend out for 'self.criterion = nn.CrossEntropyLoss()'
        return final_out
