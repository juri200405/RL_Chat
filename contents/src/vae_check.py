import argparse
from pathlib import Path
import random
import pickle

import torch
import torch.nn as nn

import tqdm

import sentencepiece as spm

from transformers import BertModel

from bert_dataloader import get_dataloader
from encoder_decoder import transformer_Decoder, transformer_Encoder, Transformer_Embedding
from config import Config


class VAE_tester:
    def __init__(self, config, sp):
        self.sp = sp
        self.config = config

        embedding = Transformer_Embedding(config)
        self.encoder = transformer_Encoder(config, embedding, nn.LayerNorm(config.d_model))
        self.decoder = transformer_Decoder(config, embedding, nn.LayerNorm(config.d_model))

    def load_pt(self, pt_file)
        checkpoint = torch.load(args.pt_file, map_location="cpu")
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, sentence):
        data = self.sp.encode(sentence)
        with torch.no_grad():
            input_s = torch.tensor([1] + data + [2]).unsqueeze(0)
            inp_mask = torch.tensor([[False]*input_s.shape[1] + [True]*(self.config.max_len - input_s.shape[1])])
            pad = torch.full((1, self.config.max_len - input_s.shape[1]), 3, dtype=torch.long)
            input_s = torch.cat((input_s, pad), dim=1)
            memory = self.encoder(input_s, attention_mask=inp_mask)
        return memory

    def generate(self, memory):
        with torch.no_grad():
            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long)  # <s>
            # tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long)
            unfinish = torch.ones(memory.shape[0], 1, dtype=torch.long)
            while tgt.shape[1] <= self.config.max_len:
                out = self.decoder(tgt, memory)
                _, topi = out.transpose(0,1).topk(1)
                next_word = topi[:,-1]
                next_word = next_word*unfinish + (3)*(1-unfinish)
                tgt = torch.cat((tgt, next_word), dim=-1)
                unfinish = unfinish * (~(next_word == 2)).long()
                if unfinish.max() == 0: # </s>
                    break
        return self.sp.decode(tgt.tolist())

    def reconstract(self, sentence):
        memory = self.encode(sentence)
        return self.generate(memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-p", "--hyper_param", required=True)
    parser.add_argument("--pt_file")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    config = Config()
    config.load_json(args.hyper_param)

    tester = VAE_tester(config, sp)
    tester.load_pt(args.pt_file)

    with open(args.input_file, 'rb') as f:
        train_dataset = pickle.load(f)

    for epoch in range(20):
        input_sentence = sp.decode(random.choice(train_dataset))
        output_sentence = tester.reconstract(input_sentence)
        output_text = "\n{}\n-> {}\n".format(input_sentence, output_sentence)
        print(output_text)
