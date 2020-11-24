import math
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from transformers import BertModel
import pytorch_lightning as pl

import sentencepiece as spm

from losses import MmdLoss


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

    def position_encode(self, x):
        return x + self.pe[:x.size(0), :]

    def forward(self, x):
        # x : (seq_len, batch_size)
        # pos_id = self.position_ids[:, :x.shape[0]]

        x = self.embedding(x)
        # x : (seq_len, batch_size, d_model)

        x = self.position_encode(x)
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
        # self.memory2logv = nn.Linear(config.mlp_n_hidden, config.n_latent)
        # self.tanh = nn.Tanh()
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
        # logv = self.tanh(self.memory2logv(memory))
        # logv = self.memory2logv(memory)

        # v = torch.diag_embed(logv.exp())
        # m = MultivariateNormal(mean, covariance_matrix=v)
        # z = m.rsample()
        # return m, z

        # z = self.reparameterize(mean, logv)
        # return z

        return mean

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
        # with torch.no_grad():
        #    tgt = self.embedding(tgt)
        #    tgt = tgt.transpose(0,1)
        tgt = self.embedding(tgt.transpose(0,1))

        # tgt = tgt.to(latent.device)
        tgt = tgt
        if tgt_mask is None:
            # tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(latent.device)
            tgt_mask = self.generate_square_subsequent_mask(len(tgt), tgt.device)

        memory = self.hidden2memory(self.relu(self.latent2hidden(latent)))
        # memory = self.latent2memory(latent)
        memory = torch.reshape(memory, (memory.shape[0], -1, tgt.shape[2])).transpose(0,1)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class MMD_VAE(pl.LightningModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.val_count = 0
        self.sp = spm.SentencePieceProcessor(model_file=args.spm_model)

        embedding_module = Transformer_Embedding(config)
        self.encoder = transformer_Encoder(config, embedding_module, nn.LayerNorm(config.d_model))
        self.decoder = transformer_Decoder(config, embedding_module, nn.LayerNorm(config.d_model))

        ce_weight = torch.ones(config.n_vocab)
        ce_weight[2] = args.eos_weight
        ce_weight[3] = args.pad_weight
        self.losses = MmdLoss(ce_weight, config, args.ignore_pad, reducation="mean")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--eos_weight', type=float, default=1.0)
        parser.add_argument('--pad_weight', type=float, default=0.0)
        parser.add_argument('--ignore_pad', type=bool, default=True)
        return parser

    def encode(self, inputs, inp_mask):
        return self.encoder(inputs, attention_mask=inp_mask)

    def generate(self, memory):
        tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.device) # <s>

        # </s>を出力してたら0, してなかったら1
        unfinish = torch.ones(memory.shape[0], dtype=torch.long, device=self.device)
        while tgt.shape[1] <= self.config.max_len:
            out = self.decoder(tgt, memory)
            _, topi = out.topk(1)
            next_word = topi[-1]
            # </s>を出力してたら<pad>にする
            next_word = next_word*unfinish + (3)*(1-unfinish)
            tgt = torch.cat((tgt, next_word), dim=-1)

            unfinish = unfinish * (~(next_word == 2)).long()
            if unfinish.max() == 0:
                break
        return tgt

    def forward(self, sentence, inp_mask, tgt_mask):
        inputs = sentence[:,:].detach()
        tgt = sentence[:,:].detach()
        label = sentence[:,1:].detach()
        inp_mask = tgt_mask
        tgt_mask = tgt_mask

        memory = self.encoder(inputs, attention_mask=inp_mask)
        outs = self.decoder(tgt, memory, tgt_padding_mask=tgt_mask)
        # outs = outs[:-1].contiguous().view(-1, outs.shape[-1])
        outs = outs[:-1,:].transpose(0,1).transpose(1,2)
        # label = label.transpose(0,1).contiguous().view(-1)

        return outs, label, memory

    def forward_loss(self, sentence, inp_mask, tgt_mask):
        outs, label, memory = self.forward(sentence, inp_mask, tgt_mask)
        loss, ce, mmd = self.losses.forward(outs, label, memory)
        return loss, ce, mmd

    def training_step(self, batch, batch_idx):
        sentence, inp_mask, tgt_mask = batch
        loss, ce, mmd = self.forward_loss(sentence, inp_mask, tgt_mask)

        log_loss = {"train/loss": loss.detach(), "train/ce_loss":ce.detach(), "train/mmd":mmd.detach()}
        self.log_dict(log_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sentence, inp_mask, tgt_mask = batch
        outs, label, memory = self.forward(sentence, inp_mask, tgt_mask)
        loss, ce, mmd = self.losses.forward(outs, label, memory)
        if self.global_rank == 0:
            rec = self.generate(memory)
            with open(
                    str(Path(self.logger.log_dir) / "val_text_{}.csv".format(self.val_count)),
                    "at", encoding='utf-8'
                    ) as f:
                f.write("\n".join(['"{}","{}"'.format(i,r) for i, r in zip(self.sp.decode(label.cpu().tolist()), self.sp.decode(rec.cpu().tolist()))]))
        return (loss.detach(), ce.detach(), mmd.detach())

    def validation_epoch_end(self, validation_step_outputs):
        losses = [torch.cat([i.unsqueeze(0) for i in item]).mean() for item in zip(*validation_step_outputs)]
        keys = ["val/loss", "val/ce_loss", "val/mmd"]
        self.log_dict(dict(zip(keys, losses)))
        self.val_count += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
