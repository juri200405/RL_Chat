import argparse
from pathlib import Path
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import tqdm

import sentencepiece as spm

# from torchviz import make_dot

from bert_dataloader import get_dataloader
from encoder_decoder import transformer_Decoder, Transformer_Embedding, transformer_Encoder
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
        print("device = {}".format(self.config.device))

        if self.config.model_type == "transformer":
            embedding_model = Transformer_Embedding(self.config)
            self.encoder = transformer_Encoder(self.config, embedding_model, nn.LayerNorm(self.config.d_model))
            # self.encoder = transformer_Encoder(self.config, Transformer_Embedding(self.config), nn.LayerNorm(self.config.d_model))
        else:
            print("model_type missmatch")
            exit()

        with open(args.input_file, 'rb') as f:
            dataset = pickle.load(f)
        val_size = 32 * self.config.batch_size * self.config.accumulate_size
        train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_size, val_size])

        self.train_dataloader = get_dataloader(train_dataset, self.config.batch_size, pad_index=3, bos_index=1, eos_index=2, fix_len = self.config.max_len)
        self.val_dataloader = get_dataloader(self.val_dataset, self.config.batch_size, pad_index=3, bos_index=1, eos_index=2, fix_len = self.config.max_len, shuffle=False)

        # self.loss_func = VaeLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='sum'), self.config, len(train_dataloader)).forward
        self.loss_func = MmdLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='mean'), self.config).forward
        self.config.save_json(str(self.output_dir / "hyper_param.json"))

        self.decoder = transformer_Decoder(self.config, embedding_model, nn.LayerNorm(self.config.d_model))
        # self.decoder = transformer_Decoder(self.config, Transformer_Embedding(self.config), nn.LayerNorm(self.config.d_model))

        self.encoder.to(self.config.device)
        self.decoder.to(self.config.device)


        if self.config.optim_type == "Adam":
            self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.config.lr)
            self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=self.config.lr)
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
        self.num_val = 0
        self.out = open(str(self.log_dir / "log"), 'wt', encoding='utf-8')
        t_itr = tqdm.trange(self.config.num_epoch, leave=False, ncols=180, file=self.out)
        for epoch in t_itr:
            self.train(epoch)
            input_data, output_data, rand_sample = self.test()
            output_text = '"{}","{}"'.format(self.sp.decode(input_data), self.sp.decode(output_data))
            self.text_logger(t_itr, "epoch_output_text", output_text, epoch)

            torch.save({
                    'epoch': self.true_epoch + epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'encoder_opt_state_dict': self.encoder_opt.state_dict(),
                    'decoder_opt_state_dict': self.decoder_opt.state_dict()
                    },
                str(self.output_dir / "epoch{:03d}.pt".format(epoch)))

        self.writer.close()
        self.out.close()

    def forward(self, sentence, padding_mask, step):
        inputs = sentence[:,:]
        tgt = sentence[:,:]
        label = sentence[:,1:]

        inputs = inputs.to(self.config.device)
        tgt = tgt.to(self.config.device)
        label = label.to(self.config.device)
        padding_mask = padding_mask.to(self.config.device)

        # m, memory = encoder(inputs, attention_mask=padding_mask)
        memory = self.encoder(inputs, attention_mask=padding_mask)
        out = self.decoder(tgt, memory, tgt_padding_mask=padding_mask)
        # make_dot(out).render(str(self.output_dir + "graph"))

        out = out[:-1].contiguous().view(-1, out.shape[-1])
        label = label.transpose(0,1).contiguous().view(-1)

        # loss, cross_entropy, kl_loss, kl_weight
        # return self.loss_func(out, label, m, step)

        loss, cross_entropy, mmd = self.loss_func(out, label, memory)
        return loss, cross_entropy, mmd, memory

    def train_process(self, sentence, padding_mask, step):
        loss, cross_entropy, mmd, _ = self.forward(sentence, padding_mask, step)
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

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        train_itr = tqdm.tqdm(self.train_dataloader, leave=False, ncols=180, file=self.out)
        n = epoch * len(train_itr)
        for sentence, padding_mask in train_itr:
            loss_item, cross_entropy_item, mmd_item = self.train_process(sentence, padding_mask, n)
            # train_itr.set_postfix({"loss":loss.item(), "ce_loss":cross_entropy.item() , "weight":kl_weight, "kl_loss":kl_loss.item()})
            train_itr.set_postfix({"loss":loss_item, "ce_loss":cross_entropy_item , "mmd":mmd_item})
            self.writer.add_scalar('train/loss',loss_item, n)
            self.writer.add_scalar('train/cross_entropy', cross_entropy_item, n)
            # self.writer.add_scalar('train/kl_loss', kl_loss.item(), n)
            # self.writer.add_scalar('train/kl_weight', kl_weight, n)
            self.writer.add_scalar('train/mmd', mmd_item, n)

            if n % self.config.log_interval == 0:
                torch.save({
                        'epoch': self.true_epoch + epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'encoder_opt_state_dict': self.encoder_opt.state_dict(),
                        'decoder_opt_state_dict': self.decoder_opt.state_dict()
                        },
                    str(self.output_dir / "{:04d}k.pt".format(n//1000)))

                input_ids, ids, rand_ids = self.test()
                output_text = '"{}","{}"'.format(self.sp.decode(input_ids), self.sp.decode(ids))
                self.text_logger(train_itr, "step_output_text", output_text, n)
                self.text_logger(train_itr, "random_output_text", self.sp.decode(rand_ids), n)

                self.encoder.train()
                self.decoder.train()

            n += 1

    def test(self):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            val_itr = tqdm.tqdm(self.val_dataloader, leave=False, ncols=180, file=self.out)
            loss_items = []
            ce_items = []
            mmd_items = []
            memorys = []
            sentences = []
            for sentence, padding_mask in val_itr:
                sentences.append(sentence.detach())
                loss, cross_entropy, mmd, memory = self.forward(sentence, padding_mask, self.num_val)

                loss_items.append(loss.item())
                ce_items.append(cross_entropy.item())
                mmd_items.append(mmd.item())
                memorys.append(memory.cpu().detach())
                cross_entropy = None
                mmd = None
                loss = None
            self.writer.add_scalar('val/loss',np.mean(loss_items), self.num_val)
            self.writer.add_scalar('val/cross_entropy',np.mean(ce_items), self.num_val)
            self.writer.add_scalar('val/mmd',np.mean(mmd_items), self.num_val)

            memorys = torch.cat(memorys, dim=0)
            memorys = torch.cat((memorys, torch.randn(1024, self.config.n_latent)), dim=0)
            sentences = self.sp.decode(torch.cat(sentences, dim=0).tolist())
            sentences += (["<RAND>"] * 1024)
            self.writer.add_embedding(memorys, metadata=sentences, global_step=self.num_val)

            data = random.choice(self.val_dataset)
            input_s = torch.tensor([1] + data + [2], device=self.config.device).unsqueeze(0)
            inp_mask = torch.tensor([[False]*input_s.shape[1] + [True]*(self.config.max_len - input_s.shape[1])], device=self.config.device)
            pad = torch.full((1, self.config.max_len - input_s.shape[1]), 3, dtype=torch.long, device=self.config.device)
            input_s = torch.cat((input_s, pad), dim=1)
            # _, memory = self.encoder(input_s, attention_mask=inp_mask)
            memory = self.encoder(input_s, attention_mask=inp_mask)

        ids = self.generate_sentence(memory)
        rand_ids = self.generate_sentence(torch.randn(1, self.config.n_latent, device=self.config.device))

        self.num_val += 1
        return data, ids[0], rand_ids[0]

    def generate_sentence(self, memory):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.config.device)  # <s>
            unfinish = torch.ones(memory.shape[0], 1, dtype=torch.long, device=self.config.device)
            while tgt.shape[1] <= self.config.max_len:
                out = self.decoder(tgt, memory)
                _, topi = out.transpose(0,1).topk(1)
                next_word = topi[:,-1]
                next_word = next_word*unfinish + (3)*(1-unfinish)
                tgt = torch.cat((tgt, next_word), dim=-1)
                unfinish = unfinish * (~(next_word == 2)).long()
                if unfinish.max() == 0: # </s>
                    break
        return tgt.cpu().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-l", "--log_dir", required=True)
    parser.add_argument("-p", "--hyper_param", required=True)
    parser.add_argument("--pt_file")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
