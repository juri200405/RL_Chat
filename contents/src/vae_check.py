import argparse
from pathlib import Path
import random
import json
from itertools import chain
import sys
import decimal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torch_optimizer

import numpy as np
import tqdm

import sentencepiece as spm

from transformers import BertModel

from bert_data import txt_to_idlist
from bert_dataloader import get_dataloader
from encoder_decoder import Bert_Encoder_gru, Bert_Encoder_vae, transformer_Decoder

def test(encoder, decoder, test_data, loss_func, n_vocab, encoder_device, decoder_device, max_length=128, vae=False):
    encoder.eval()
    decoder.eval()
    data = random.choice(test_data)

    input_s = torch.tensor([1] + data + [2], device=encoder_device).unsqueeze(0)
    if vae:
        _, _, memory = encoder(input_s)
    else:
        memory = encoder(input_s)
    memory = memory.to(decoder_device)

    tgt = torch.full((1, max_length), 0, dtype=torch.long, device=encoder_device)
    tgt_key_padding_mask = torch.full((1, max_length), True, dtype=torch.bool, device=decoder_device)
    t_word = 1 # <s>
    ids = []
    for t in range(max_length):
        tgt[0][t] = t_word
        tgt_key_padding_mask[0][t] = False
        out = decoder(tgt, memory, tgt_padding_mask=tgt_key_padding_mask)
        _, topi = out.topk(1)
        next_word = topi[t].item()
        ids.append(next_word)
        if next_word == 2: # </s>
            break
        else:
            t_word = next_word
    return data, ids

def anneal_function(step, x0, k=0.00002):
    tmp = 1/(1+decimal.Decimal(-k*(step-x0)).exp())
    # print("tmp:{}, < min:{}, step:{}, x0:{}".format(tmp, tmp<sys.float_info.min, step, x0))
    if tmp < sys.float_info.min:
        tmp = sys.float_info.min
    return float(tmp)

def get_vae_loss(label_loss_func, decoder_device, batch_size):
    label_loss_func = label_loss_func
    def _f(out, label, logv, mean, step, x0):
        logv = logv.to(decoder_device)
        mean = mean.to(decoder_device)
        closs_entropy_loss = label_loss_func(out, label)
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = anneal_function(step, x0)
        return (closs_entropy_loss + KL_weight * KL_loss) / batch_size, closs_entropy_loss, KL_loss, KL_weight
    return _f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-p", "--hyper_param", required=True)
    parser.add_argument("-b", "--bert_path", required=True)
    parser.add_argument("--pt_file")
    args = parser.parse_args()


    with open(args.hyper_param, 'rt') as f:
        hyperp = json.load(f)

    # n_vocab = hyperp["n_vocab"]
    # d_model = hyperp["d_model"]
    n_hidden = hyperp["n_hidden"]
    decoder_nlayers = hyperp["decoder_nlayers"]
    n_head = hyperp["n_head"]
    dropout = hyperp["dropout"]
    use_gpus = hyperp["use_gpus"]
    model_type = hyperp["model_type"]
    batch_size = hyperp["batch_size"]
    optim_type = hyperp["optim_type"]

    # available_cuda = torch.cuda.is_available()
    # encoder_device = decoder_device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    # decoder_device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    # encoder_device = torch.device('cpu')
    device_num = torch.cuda.device_count()
    if len(use_gpus) > 1:
        if device_num > 1:
            encoder_device = torch.device('cuda', use_gpus[0])
            decoder_device = torch.device('cuda', use_gpus[1])
            device_name = "cuda"
        elif device_num == 1:
            encoder_device = decoder_device = torch.device('cuda', use_gpus[0])
            device_name = "cuda"
        else:
            encoder_device = decoder_device = torch.device('cpu')
            device_name = "cpu"
    elif len(use_gpus) == 1:
        if device_num >= 1:
            encoder_device = decoder_device = torch.device('cuda', usegpus[0])
            device_name = "cuda"
        else:
            encoder_device = decoder_device = torch.device('cpu')
            device_name = "cpu"
    else:
        encoder_device = decoder_device = torch.device('cpu')
        device_name = "cpu"

    print("encoder = {}, decoder = {}".format(encoder_device, decoder_device))


    if model_type == "gru":
        encoder = Bert_Encoder_gru(args.bert_path)
        loss_func = nn.CrossEntropyLoss(ignore_index=3)
        vae = False
    elif model_type == "vae":
        encoder = Bert_Encoder_vae(args.bert_path)
        loss_func = get_vae_loss(nn.CrossEntropyLoss(ignore_index=3, reduction='sum'), decoder_device, batch_size)
        vae = True
    else:
        print("model_type missmatch")
        exit()
    n_vocab = encoder.bert.config.vocab_size
    d_model = encoder.bert.config.hidden_size


    hyperp["n_vocab"] = n_vocab
    hyperp["d_model"] = d_model
    # hyperp["spm_model"] = args.spm_model
    # hyperp["input_file"] = args.input_file
    # hyperp["bert_path"] = args.bert_path

    decoder = transformer_Decoder(n_vocab, d_model, n_head, n_hidden, decoder_nlayers, encoder.bert.get_input_embeddings(), nn.LayerNorm(d_model), dropout=dropout)

    # encoderのBERT内に組み込まれてる BertEmbeddings をdecoderで使うため、GPUへ送る順番は decoder->encoder
    decoder = decoder.to(decoder_device)
    encoder = encoder.to(encoder_device)


    if optim_type == "Adam":
        encoder_opt = optim.Adam(encoder.parameters())
        decoder_opt = optim.Adam(decoder.parameters())
    elif optim_type == "RAdam":
        encoder_opt = torch_optimizer.RAdam(encoder.parameters())
        decoder_opt = torch_optimizer.RAdam(decoder.parameters())
    elif optim_type == "Yogi":
        encoder_opt = torch_optimizer.Yogi(encoder.parameters())
        decoder_opt = torch_optimizer.Yogi(decoder.parameters())
    else:
        print("optim_type missmatch")
        exit()

    if args.pt_file is not None:
        checkpoint = torch.load(args.pt_file)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        encoder_opt.load_state_dict(checkpoint["encoder_opt_state_dict"])
        decoder_opt.load_state_dict(checkpoint["decoder_opt_state_dict"])
        true_epoch = checkpoint["epoch"] + 1
    else:
        true_epoch = 0

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    train_dataset = txt_to_idlist(sp, args.input_file, 3)

    for epoch in range(20):
        input_data, output_data = test(encoder, decoder, train_dataset, loss_func, n_vocab, encoder_device, decoder_device, vae=vae)
        output_text = "{}\n{}\n-> {}\n   {}".format(sp.decode(input_data), input_data, sp.decode(output_data), output_data)
        print(output_text)

    writer.close()
