import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from transformers import BertModel


class Transformer_Embedding(nn.Module):
    def __init__(self, config):
        super(Transformer_Embedding, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

        pe = torch.zeros(config.max_len, config.d_model)
        position = torch.arange(0, config.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # self.position_embedding = nn.Embedding(max_len, d_model)
        # self.register_buffer("position_ids", torch.arange(max_len).expand((1,-1)))

    def forward(self, x):
        # x : (seq_len, batch_size)
        # pos_id = self.position_ids[:, :x.shape[0]]

        x = self.embedding(x)
        # x : (seq_len, batch_size, d_model)

        x = x + self.pe[:x.size(0), :]
        # x = x + self.position_embedding(pos_id).transpose(0,1)
        # x : (seq_len, batch_size, d_model)
        return self.dropout(x)


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
        hidden = torch.zeros(1, bert_output.shape[0], self.gru.hidden_size, device=bert_output.device)
        _, memory = self.gru(bert_output, hidden)
        mean = self.memory2mean(memory)
        logv = self.memory2logv(memory)

        m = MultivariateNormal(mean, torch.diag_embed(logv.exp()))
        z = m.rsample()
        return m, z # (1, batch, d_model)

class transformer_Encoder(nn.Module):
    def __init__(self, config, embedding, norm=None):
        super(transformer_Encoder, self).__init__()
        self.embedding = embedding
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(config.d_model, config.n_head, config.n_hidden, config.dropout), config.encoder_nlayers, norm)
        
        self.fc = nn.Linear(config.max_len * config.d_model, config.mlp_n_hidden)
        self.memory2mean = nn.Linear(config.mlp_n_hidden, config.n_latent)
        self.memory2logv = nn.Linear(config.mlp_n_hidden, config.n_latent)
        self.tanh = nn.Tanh()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, src, attention_mask=None):
        # src : (batch_size, seq_len)

        src = self.embedding(src.transpose(0,1))
        # src : (seq_len, batch_size, d_model)

        out = self.transformer_encoder(src, src_key_padding_mask=attention_mask)
        # out = (seq_len, batch_size, d_model)
        out = torch.reshape(out.transpose(0,1), (out.shape[1], -1))

        memory = F.relu(self.fc(out))
        mean = self.memory2mean(memory)
        # logv = self.tanh(self.memory2logv(memory))
        logv = self.memory2logv(memory)

        # v = torch.diag_embed(logv.exp())
        # m = MultivariateNormal(mean, covariance_matrix=v)
        # z = m.rsample()
        # return m, z
        z = self.reparameterize(mean, logv)
        return z

class transformer_Decoder(nn.Module):
    def __init__(self, config, embedding, norm=None):
        super(transformer_Decoder, self).__init__()
        self.embedding = embedding
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(config.d_model, config.n_head, config.n_hidden, config.dropout), config.decoder_nlayers, norm)
        self.fc = nn.Linear(config.d_model, config.n_vocab)

        self.latent2hidden = nn.Linear(config.n_latent, config.mlp_n_hidden)
        self.hidden2memory = nn.Linear(config.mlp_n_hidden, config.max_len * config.d_model)

    def forward(self, tgt, latent, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        # with torch.no_grad():
        #    tgt = self.embedding(tgt)
        #    tgt = tgt.transpose(0,1)
        tgt = self.embedding(tgt.transpose(0,1))

        tgt = tgt.to(latent.device)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(latent.device)

        memory = self.hidden2memory(F.relu(self.latent2hidden(latent)))
        # memory = self.latent2memory(latent)
        memory = torch.reshape(memory, (tgt.shape[1], tgt.shape[0], tgt.shape[2])).transpose(0,1)

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
