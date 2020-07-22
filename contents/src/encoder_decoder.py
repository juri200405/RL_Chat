import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from transformers import BertModel


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
