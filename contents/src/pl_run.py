            # weights_summary='full'1
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
import pytorch_lightning as pl

from bert_dataloader import get_dataloader
from encoder_decoder import Bert_Encoder_vae, transformer_Decoder, Transformer_Embedding, transformer_Encoder, MMD_VAE
from config import Config
from losses import VaeLoss, MmdLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-l", "--log_dir", required=True)
    parser.add_argument("-p", "--hyper_param", required=True)
    parser.add_argument("-b", "--bert_path", required=True)
    parser.add_argument("--pt_file")
    parser = MMD_VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    # writer = SummaryWriter(log_dir=args.log_dir)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    config = Config()
    config.load_json(args.hyper_param)
    config.n_vocab = len(sp)
    config.save_json(str(output_dir / "hyper_param.json"))

    with open(args.input_file, 'rb') as f:
        dataset = pickle.load(f)
    val_size = 20 * config.batch_size * args.gpus
    test_size = 20 * config.batch_size * args.gpus
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_size-test_size, val_size, test_size])
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [len(dataset)-val_size-test_size, val_size, test_size])
    # train_dataset, _, _ = torch.utils.data.random_split(dataset, [len(dataset)-val_size-test_size, val_size, test_size])
    train_dataloader = get_dataloader(
            train_dataset,
            config.batch_size,
            pad_index=3, bos_index=1, eos_index=2,
            fix_len = config.max_len,
            num_workers=2, shuffle=True
            )
    val_dataloader = get_dataloader(
            val_dataset,
            config.batch_size,
            pad_index=3, bos_index=1, eos_index=2,
            fix_len = config.max_len,
            num_workers=2, shuffle=False
            )

    model = MMD_VAE(config, args)
    trainer = pl.Trainer.from_argparse_args(
            args,
            default_root_dir=args.log_dir,
            weights_save_path=args.output_dir,
            accelerator='ddp',
            benchmark=True,
            log_every_n_steps=10,
            flush_logs_every_n_steps=50,
            accumulate_grad_batches=config.accumulate_size
            )
    trainer.fit(model, train_dataloader, val_dataloader)
    # trainer.fit(model, train_dataloader)
