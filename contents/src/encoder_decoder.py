import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

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

    def position_encode(self, x):
        return x + self.pe[:x.size(0), :]

    def forward(self, x):
        # x : (seq_len, batch_size)

        x = self.embedding(x)
        # x : (seq_len, batch_size, d_model)

        x = self.position_encode(x)
        # x : (seq_len, batch_size, d_model)
        return self.dropout(x)


class transformer_Encoder(nn.Module):
    def __init__(self, config, embedding, norm=None):
        super(transformer_Encoder, self).__init__()
        self.embedding = embedding
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(config.d_model, config.n_head, config.n_hidden, config.dropout), config.encoder_nlayers, norm)
        
        self.fc = nn.Linear(config.max_len * config.d_model, config.mlp_n_hidden)
        self.memory2mean = nn.Linear(config.mlp_n_hidden, config.n_latent)
        self.memory2logv = nn.Linear(config.mlp_n_hidden, config.n_latent)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

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

        memory = self.relu(self.fc(out))
        mean = self.memory2mean(memory)

        # return mean

        # logv = self.tanh(self.memory2logv(memory))
        logv = self.memory2logv(memory)

        # v = torch.diag_embed(logv.exp())
        # m = MultivariateNormal(mean, covariance_matrix=v)
        # z = m.rsample()
        # return m, z

        z = self.reparameterize(mean, logv)
        return self.tanh(z)

class transformer_Decoder(nn.Module):
    def __init__(self, config, embedding, norm=None):
        super(transformer_Decoder, self).__init__()
        self.embedding = embedding
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(config.d_model, config.n_head, config.n_hidden, config.dropout), config.decoder_nlayers, norm)
        self.fc = nn.Linear(config.d_model, config.n_vocab)

        self.latent2hidden = nn.Linear(config.n_latent, config.mlp_n_hidden)
        self.hidden2memory = nn.Linear(config.mlp_n_hidden, config.max_len * config.d_model)
        self.relu = nn.LeakyReLU()

    def forward(self, tgt, latent, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        tgt = self.embedding(tgt.transpose(0,1))

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(latent.device)

        memory = self.hidden2memory(self.relu(self.latent2hidden(latent)))
        memory = torch.reshape(memory, (memory.shape[0], -1, tgt.shape[2])).transpose(0,1)

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
