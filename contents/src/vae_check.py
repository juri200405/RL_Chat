import argparse
from pathlib import Path
import random
import pickle

import torch
import torch.nn as nn

import tqdm

import sentencepiece as spm
import transformers

from bert_dataloader import get_dataloader
from encoder_decoder import transformer_Decoder, transformer_Encoder, Transformer_Embedding
from config import Config


class VAE_tester:
    def __init__(self, config, sp, device):
        self.sp = sp
        self.config = config
        self.device = device

        embedding = Transformer_Embedding(config)
        self.encoder = transformer_Encoder(config, embedding, nn.LayerNorm(config.d_model))
        self.decoder = transformer_Decoder(config, embedding, nn.LayerNorm(config.d_model))
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def load_pt(self, pt_file):
        checkpoint = torch.load(pt_file, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, sentence):
        data = self.sp.encode(sentence)[:self.config.max_len-2]
        return self.encode_from_ids(data)

    def encode_from_ids(self, ids):
        with torch.no_grad():
            input_s = torch.tensor([1] + ids + [2]).unsqueeze(0)
            inp_mask = torch.tensor([[False]*input_s.shape[1] + [True]*(self.config.max_len - input_s.shape[1])])
            pad = torch.full((1, self.config.max_len - input_s.shape[1]), 3, dtype=torch.long)
            input_s = torch.cat((input_s, pad), dim=1)
            memory = self.encoder(input_s.to(self.device), attention_mask=inp_mask.to(self.device))
        return memory

    def generate_ids(self, memory):
        with torch.no_grad():
            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.device)  # <s>
            unfinish = torch.ones(memory.shape[0], 1, dtype=torch.long, device=self.device)
            memory = memory.to(self.device)
            while tgt.shape[1] < self.config.max_len-1:
                out = self.decoder(tgt, memory)
                _, topi = out.transpose(0,1).topk(1)
                next_word = topi[:,-1]
                next_word = next_word*unfinish + (3)*(1-unfinish)
                tgt = torch.cat((tgt, next_word), dim=-1)
                unfinish = unfinish * (~(next_word == 2)).long()
                if unfinish.max() == 0: # </s>
                    break
        return tgt.cpu().tolist()

    def generate(self, memory):
        ids = self.generate_ids(memory)
        return self.sp.decode(ids)

    def beam_generate_ids(self, memory, num_beams):
        batch_size = memory.shape[0]
        vocab_size = self.config.n_vocab
        beam_scorer = transformers.BeamSearchScorer(batch_size=batch_size, max_length=self.config.max_len, num_beams=num_beams, device=self.device)
        with torch.no_grad():
            memory_list = torch.split(memory, 1)
            memory_list = [torch.cat([item]*num_beams) for item in memory_list]
            memory = torch.cat(memory_list)
            memory = memory.to(self.device)

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
            beam_scores[:,1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.device)  # <s>
            while tgt.shape[1] < self.config.max_len-1:
                out = self.decoder(tgt, memory) # (seq, batch, n_vocab)
                out = out.transpose(0,1)[:, -1, :]

                next_token_scores = torch.nn.functional.log_softmax(out, dim=-1)
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(next_token_scores, 2*num_beams, dim=1, largest=True, sorted=True)
                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(tgt, next_token_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                tgt = torch.cat((tgt[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1)

                if beam_scorer.is_done:
                    break
            decoded = beam_scorer.finalize(tgt, beam_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
        return decoded.cpu().tolist()

    def beam_generate(self, memory, num_beams):
        ids = self.beam_generate_ids(memory, num_beams)
        return self.sp.decode(ids)

    def reconstract(self, sentence):
        memory = self.encode(sentence)
        return self.generate(memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("--pt_file")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    hyper_param = Path(args.pt_file).with_name("hyper_param.json")
    config = Config()
    config.load_json(str(hyper_param))

    tester = VAE_tester(config, sp, "cuda:2")
    tester.load_pt(args.pt_file)

    with open(args.input_file, 'rb') as f:
        train_dataset = pickle.load(f)

    for epoch in range(20):
        input_sentence = sp.decode(random.choice(train_dataset))
        output_sentence = tester.reconstract(input_sentence)
        output_text = "\n{}\n-> {}\n".format(input_sentence, output_sentence)
        print(output_text)

    memorys = torch.randn(20, config.n_latent).tanh()
    print("\n".join(tester.generate(memorys)))
