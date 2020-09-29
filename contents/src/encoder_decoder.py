import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from transformers import BertModel


class Transformer_Embedding(nn.Module):
    def __init__(self, n_vocab, d_model, dropout, max_len):
        super(Transformer_Embedding, self).__init__()

        self.embedding = nn.Embedding(n_vocab, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : (seq_len, batch_size)

        x = self.embedding(x)
        # x : (seq_len, batch_size, d_model)

        x = x + self.pe[:x.size(0), :]
        # x : (seq_len, batch_size, d_model)
        return self.dropout(x)

class Bert_Encoder_gru(nn.Module):
    def __init__(self, bert_path):
        super(Bert_Encoder_gru, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size, batch_first=True)


    def forward(self, src, attention_mask=None):
        with torch.no_grad():
            bert_output = self.bert(src, attention_mask=attention_mask)
            bert_output = bert_output[0]
        # bertの最終隠れ層を、GRUで合成したものを出力
        hidden = torch.zeros(1, bert_output.size()[0], self.gru.hidden_size, device=bert_output.device)
        _, memory = self.gru(bert_output, hidden)
        return memory # (1, batch, d_model)

class Bert_Encoder_vae(nn.Module):
    def __init__(self, bert_path):
        super(Bert_Encoder_vae, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.memory2mean = nn.Linear(hidden_size, hidden_size)
        self.memory2logv = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, attention_mask=None):
        with torch.no_grad():
            bert_output = self.bert(src, attention_mask=attention_mask)
            bert_output = bert_output[0]
        # bertの最終隠れ層を、GRUで合成したものを出力
        hidden = torch.zeros(1, bert_output.size()[0], self.gru.hidden_size, device=bert_output.device)
        _, memory = self.gru(bert_output, hidden)
        mean = self.memory2mean(memory)
        logv = self.memory2logv(memory)

        m = MultivariateNormal(mean, torch.diag_embed(logv.exp()))
        z = m.rsample()
        return mean, logv, z # (1, batch, d_model)

class transformer_Encoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_head, n_hidden, n_layers, embedding, norm=None, dropout=0.1):
        super(transformer_Encoder, self).__init__()
        self.embedding = embedding
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, n_hidden, dropout), n_layers, norm)
        
        self.memory2mean = nn.Linear(d_model, d_model)
        self.memory2logv = nn.Linear(d_model, d_model)

    def forward(self, src, attention_mask=None):
        # src : (batch_size, seq_len)

        src = self.embedding(src).transpose(0,1)
        # src : (seq_len, batch_size, d_model)

        attention_mask = attention_mask.byte()
        memory = self.transformer_encoder(src, src_key_padding_mask=attention_mask)
        # memory = (seq_len, batch_size, d_model)

        mean = self.memory2mean(memory)
        logv = self.memory2logv(memory)

        m = MultivariateNormal(mean, torch.diag_embed(logv.exp()))
        z = m.rsample()
        return mean, logv, z

class transformer_Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_head, n_hidden, n_layers, embedding, norm=None, dropout=0.1):
        super(transformer_Decoder, self).__init__()
        self.embedding = embedding
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_head, n_hidden, dropout), n_layers, norm)
        self.fc = nn.Linear(d_model, n_vocab)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        with torch.no_grad():
            tgt = self.embedding(tgt)
            tgt = tgt.transpose(0,1)

        tgt = tgt.to(memory.device)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(memory.device)
        memory = memory
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
