import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from transformers import BertModel
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel, register_model
from fairseq.criterions import CrossEntropyCriterion, register_criterion


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

class Bert_Encoder_vae(FairseqEncoder):
    def __init__(self, bert_path, dictionary):
        super(Bert_Encoder_vae, self).__init__(dictionary)

        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.memory2mean = nn.Linear(hidden_size, hidden_size)
        self.memory2logv = nn.Linear(hidden_size, hidden_size)

    def forward(self, src_tokens, src_length, attention_mask=None):
        with torch.no_grad():
            bert_output = self.bert(src_tokens, attention_mask=attention_mask)
            bert_output = bert_output[0]
        # bertの最終隠れ層を、GRUで合成したものを出力
        hidden = torch.zeros(1, bert_output.size()[0], self.gru.hidden_size, device=bert_output.device)
        _, memory = self.gru(bert_output, hidden)
        mean = self.memory2mean(memory)
        logv = self.memory2logv(memory)

        mean = mean.squeeze(0)
        logv = logv.squeeze(0)
        m = MultivariateNormal(mean, torch.diag_embed(logv.exp()))
        z = m.rsample()
        return {
                'mean': mean,
                'logv': logv,
                'z': z, # (batch, d_model)
                }

    def reorder_encoder_out(self, encoder_out, new_order):
        z = encoder_out['z']
        return {
                'z': z.index_select(0, new_order)
                }

class transformer_Decoder(FairseqDecoder):
    def __init__(self, d_model, n_head, n_hidden, n_layers, embedding, dictionary, norm=None, dropout=0.1):
        super(transformer_Decoder, self).__init__(dictionary)
        n_vocab = len(dictionary)
        self.embedding = embedding
        self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model, n_head, n_hidden, dropout),
                n_layers,
                norm
                )
        self.fc = nn.Linear(d_model, n_vocab)

    def forward(self, prev_output_tokens, encoder_out, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        with torch.no_grad():
            tgt = self.embedding(prev_output_token)
            tgt = tgt.transpose(0,1) # (batch, seq, d_model) -> (seq, batch, d_model)

        tgt = tgt.to(memory.device)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(memory.device)
        memory = encoder_out.unsqueeze(0) # (batch, d_model) -> (1, batch, d_model)
        output = self.transformer_decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=memory_padding_mask
                )
        output = self.fc(output)
        return output, None

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

@register_model('bert_vae')
class Vae_encoder_decoder_Model(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--bert_path', type=str, required=True)
        parser.add_argument('--d_model', type=int, default=1024)
        parser.add_argument('--n_head', type=int, default=2)
        parser.add_argument('--n_hidden', type=int, default=2048)
        parser.add_argument('--decoder_nlayers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.0)

    @classmethod
    def build_model(cls, args, task):
        encoder = Bert_Encoder_vae(
                bert_path=args.bert_path,
                dictionary=task.source_dictionary,
                )
        decoder = transformer_Decoder(
                d_model=args.d_model,
                n_head=args.n_head,
                n_hidden=args.n_hidden,
                n_layers=args.decoder_nlayers,
                embedding=encoder.bert.get_input_embeddings(),
                dictionary=task.target_dictionaly,
                norm=nn.LayerNorm(d_model),
                dropout=args.dropout,
                )
        model = Vae_encoder_decoder_Model(encoder, decoder)
        print(model)
        return model

@register_criterion('vae_loss')
class Vae_loss(CrossEntropyCriterion):
    def __init__(self, task, sentense_avg):
        super().__init__(task, sentense_avg)

    def forward(self, model, sample, reduce=False):
        net_output = model(**sample['net_input'])
        loss

    def compute_loss():
        pass

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
