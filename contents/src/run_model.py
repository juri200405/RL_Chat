import argparse
from pathlib import Path
import random
import json
from itertools import chain
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torch_optimizer

import numpy as np
import tqdm

import sentencepiece as spm

from transformers import BertModel
# from torchviz import make_dot

from bert_dataloader import get_dataloader
from encoder_decoder import Bert_Encoder_vae, transformer_Decoder, Transformer_Embedding, transformer_Encoder
from config import Config
from losses import VaeLoss, MmdLoss


class Trainer:
    def __init__(self, args):
        self.output_dir = Path(args.output_dir)
        self.log_dir = Path(args.log_dir)
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.sp = spm.SentencePieceProcessor(model_file=args.spm_model)

        self.config = Config()
        self.config.load_json(args.hyper_param)
        self.config.n_vocab = len(self.sp)

        print(args.output_dir)
        print("encoder = {}, decoder = {}".format(self.config.encoder_device, self.config.decoder_device))

        if self.config.model_type == "bert":
            self.encoder = Bert_Encoder_vae(args.bert_path)
            self.config.d_model = self.encoder.bert.config.hidden_size
            embedding_model = self.encoder.bert.get_input_embeddings()
        elif self.config.model_type == "transformer":
            embedding_model = Transformer_Embedding(self.config)
            self.encoder = transformer_Encoder(self.config, embedding_model, nn.LayerNorm(self.config.d_model))
            # self.encoder = transformer_Encoder(self.config, Transformer_Embedding(self.config), nn.LayerNorm(self.config.d_model))
        else:
            print("model_type missmatch")
            exit()

        with open(args.input_file, 'rb') as f:
            self.train_dataset = pickle.load(f)

        self.train_dataloader = get_dataloader(self.train_dataset, self.config.batch_size, pad_index=3, bos_index=1, eos_index=2, fix_len = self.config.max_len)

        # self.loss_func = VaeLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='sum'), self.config, len(train_dataloader)).forward
        # self.loss_func = MmdLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='sum'), self.config).forward
        # self.loss_func = MmdLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='mean'), self.config).forward
        cross_entropy_weight = torch.ones(self.config.n_vocab, device=self.config.decoder_device)
        cross_entropy_weight[2] = 20
        cross_entropy_weight[3] = 0.05
        # self.loss_func = MmdLoss(nn.CrossEntropyLoss(weight=cross_entropy_weight, ignore_index=3, reduction='mean'), self.config).forward
        self.loss_func = MmdLoss(nn.CrossEntropyLoss(weight=cross_entropy_weight, reduction='mean'), self.config).forward
        self.config.save_json(str(self.output_dir / "hyper_param.json"))

        self.decoder = transformer_Decoder(self.config, embedding_model, nn.LayerNorm(self.config.d_model))
        # self.decoder = transformer_Decoder(self.config, Transformer_Embedding(self.config), nn.LayerNorm(self.config.d_model))

        # encoderのBERT内に組み込まれてる BertEmbeddings をdecoderで使うため、GPUへ送る順番は decoder->encoder
        self.decoder.to(self.config.decoder_device)
        self.encoder.to(self.config.encoder_device)


        if self.config.optim_type == "Adam":
            self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.config.lr)
            self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=self.config.lr)
        elif self.config.optim_type == "RAdam":
            self.encoder_opt = torch_optimizer.RAdam(self.encoder.parameters(), lr=self.config.lr)
            self.decoder_opt = torch_optimizer.RAdam(self.decoder.parameters(), lr=self.config.lr)
        elif self.config.optim_type == "Yogi":
            self.encoder_opt = torch_optimizer.Yogi(self.encoder.parameters())
            self.decoder_opt = torch_optimizer.Yogi(self.decoder.parameters())
        else:
            print("optim_type missmatch")
            exit()

        if args.pt_file is not None:
            checkpoint = torch.load(args.pt_file)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            self.encoder_opt.load_state_dict(checkpoint["encoder_opt_state_dict"])
            self.decoder_opt.load_state_dict(checkpoint["decoder_opt_state_dict"])
            self.true_epoch = checkpoint["epoch"] + 1
        else:
            self.true_epoch = 0

        with open(str(self.output_dir / "epoch_out_text.csv"), 'w', encoding='utf-8') as out:
            out.write("input_text,reconstract_text")

    def text_logger(self, itr, name, text, n):
        self.writer.add_text(name, text, n)
        itr.write(text)
        with open(str(self.output_dir / (name + ".csv")), 'at', encoding='utf-8') as f:
            f.write("\n{}".format(text))

    def run(self):
        self.out = open(str(self.log_dir / "log"), 'wt', encoding='utf-8')
        t_itr = tqdm.trange(self.config.num_epoch, leave=False, ncols=180, file=self.out)
        for epoch in t_itr:
            train_loss = self.train(scaler, epoch)
            t_itr.set_postfix({"ave_loss":train_loss})
            self.writer.add_scalar('Loss/average', train_loss, epoch)

            input_data, output_data = self.test()
            output_text = '"{}","{}"'.format(self.sp.decode(input_data), self.sp.decode(output_data))
            self.text_logger(t_itr, "epoch_output_text", output_text, epoch)

            torch.save({
                    'epoch': self.true_epoch + epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'encoder_opt_state_dict': self.encoder_opt.state_dict(),
                    'decoder_opt_state_dict': self.decoder_opt.state_dict(),
                    'train_loss': train_loss},
                str(self.output_dir / "epoch{:03d}.pt".format(epoch)))

        self.writer.close()
        self.out.close()

    def train_process(self, sentence, inp_padding_mask, tgt_padding_mask, scaler, step):
        inputs = sentence[:,:]
        tgt = sentence[:,:]
        label = sentence[:,1:]

        inputs = inputs.to(self.config.encoder_device)

        if self.config.model_type == "bert":
            inp_padding_mask = inp_padding_mask.to(self.config.encoder_device)
        else:
            inp_padding_mask = tgt_padding_mask.to(self.config.encoder_device)
        # embedding がencoderのdeviceにあるため、tgtはencoder_deviceに送る
        tgt = tgt.to(self.config.encoder_device)
        # tgt = tgt.to(self.config.decoder_device)
        label = label.to(self.config.decoder_device)
        tgt_padding_mask = tgt_padding_mask.to(self.config.decoder_device)

        # m, memory = encoder(inputs, attention_mask=inp_padding_mask)
        memory = self.encoder(inputs, attention_mask=inp_padding_mask)
        memory = memory.to(self.config.decoder_device)
        out = self.decoder(tgt, memory, tgt_padding_mask=tgt_padding_mask)
        # make_dot(out).render(str(self.output_dir + "graph"))

        out = out[:-1].contiguous().view(-1, out.shape[-1])
        label = label.transpose(0,1).contiguous().view(-1)

        # loss, cross_entropy, kl_loss, kl_weight = self.loss_func(out, label, m, step)
        loss, cross_entropy, mmd = self.loss_func(out, label, memory)
        loss = loss / self.config.accumulate_size

        loss.backward()

        if (step + 1) % self.config.accumulate_size == 0:
            self.encoder_opt.step()
            self.decoder_opt.step()

            self.encoder_opt.zero_grad()
            self.decoder_opt.zero_grad()

        loss_item = loss.item()
        cross_entropy_item = cross_entropy.item()
        mmd_item = mmd.item()
        cross_entropy = None
        mmd = None
        loss = None

        return loss_item, cross_entropy_item, mmd_item

    def train(self, scaler, epoch):
        self.encoder.train()
        self.decoder.train()
        losses = []
        train_itr = tqdm.tqdm(self.train_dataloader, leave=False, ncols=180, file=self.out)
        n = epoch * len(train_itr)
        for sentence, inp_padding_mask, tgt_padding_mask in train_itr:
        # for sentence, _ in train_itr:
            loss_item, cross_entropy_item, mmd_item = self.train_process(sentence, inp_padding_mask, tgt_padding_mask, scaler, n)
            losses.append(loss_item)
            # train_itr.set_postfix({"loss":loss.item(), "ce_loss":cross_entropy.item() , "weight":kl_weight, "kl_loss":kl_loss.item()})
            train_itr.set_postfix({"loss":loss_item, "ce_loss":cross_entropy_item , "mmd":mmd_item})
            self.writer.add_scalar('Loss/each',loss_item, n)
            self.writer.add_scalar('Detail_Loss/cross_entropy', cross_entropy_item, n)
            # self.writer.add_scalar('Detail_Loss/kl_loss', kl_loss.item(), n)
            # self.writer.add_scalar('Detail_Loss/kl_weight', kl_weight, n)
            self.writer.add_scalar('Detail_Loss/mmd', mmd_item, n)

            if n % self.config.log_interval == 0:
                torch.save({
                        'epoch': self.true_epoch + epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'encoder_opt_state_dict': self.encoder_opt.state_dict(),
                        'decoder_opt_state_dict': self.decoder_opt.state_dict(),
                        'train_loss': np.mean(losses)},
                    str(self.output_dir / "{:04d}k.pt".format(n//1000)))

                input_ids, ids = self.test()
                output_text = '"{}","{}"'.format(self.sp.decode(input_ids), self.sp.decode(ids))
                self.text_logger(train_itr, "step_output_text", output_text, n)
                rand_ids = self.generate_sentence(torch.randn(1, self.config.n_latent, device=self.config.decoder_device))
                self.text_logger(train_itr, "random_output_text", self.sp.decode(rand_ids), n)

                self.encoder.train()
                self.decoder.train()

            n += 1
        return np.mean(losses)

    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        data = random.choice(self.train_dataset)

        input_s = torch.tensor([1] + data + [2], device=self.config.encoder_device).unsqueeze(0)
        pad = torch.full((1, self.config.max_len - input_s.shape[1]), 3, dtype=torch.long, device=self.config.encoder_device)
        input_s = torch.cat((input_s, pad), dim=1)
        # _, memory = self.encoder(input_s)
        memory = self.encoder(input_s)
        memory = memory.to(self.config.decoder_device)

        ids = self.generate_sentence(memory)

        return data, ids

    def generate_sentence(self, memory):
        tgt = torch.full((1, self.config.max_len), 0, dtype=torch.long, device=self.config.encoder_device)
        # tgt = torch.full((1, self.config.max_len), 0, dtype=torch.long, device=self.config.decoder_device)
        tgt_key_padding_mask = torch.full((1, self.config.max_len), True, dtype=torch.bool, device=self.config.decoder_device)
        t_word = 1 # <s>
        ids = []
        for t in range(self.config.max_len):
            tgt[0][t] = t_word
            tgt_key_padding_mask[0][t] = False
            out = self.decoder(tgt, memory, tgt_padding_mask=tgt_key_padding_mask)
            _, topi = out.topk(1)
            next_word = topi[t].item()
            ids.append(next_word)
            if next_word == 2: # </s>
                break
            else:
                t_word = next_word
        return ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-l", "--log_dir", required=True)
    parser.add_argument("-p", "--hyper_param", required=True)
    parser.add_argument("-b", "--bert_path", required=True)
    parser.add_argument("--pt_file")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
