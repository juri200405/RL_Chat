import argparse
from pathlib import Path
import random
import json

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
from encoder_decoder import Bert_Encoder_gru, transformer_Decoder


def train(encoder, decoder, train_dataloader, loss_func, encoder_opt, decoder_opt, n_vocab, encoder_device, decoder_device, writer, epoch):
    encoder.train()
    decoder.train()
    losses = []
    train_itr = tqdm.tqdm(train_dataloader, leave=False)
    n = 0
    for sentence, inp_padding_mask, tgt_padding_mask in train_itr:
    # for sentence, _ in train_itr:
        inputs = sentence[:,:]
        tgt = sentence[:,:]
        label = sentence[:,1:]

        inputs = inputs.to(encoder_device)
        inp_padding_mask = inp_padding_mask.to(encoder_device)
        # embedding がencoderのdeviceにあるため、tgtはencoder_deviceに送る
        tgt = tgt.to(encoder_device)
        label = label.to(decoder_device)
        tgt_padding_mask = tgt_padding_mask.to(decoder_device)

        memory = encoder(inputs, attention_mask=inp_padding_mask)
        memory = memory.to(decoder_device)
        out = decoder(tgt, memory, tgt_padding_mask=tgt_padding_mask)

        out = out[:-1].contiguous().view(-1, out.shape[-1])
        label = label.transpose(0,1).contiguous().view(-1)
        loss = loss_func(out, label)

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        losses.append(loss.item())
        train_itr.set_postfix({"loss":loss.item()})
        writer.add_scalar('Loss/each',loss.item(), epoch * len(train_itr) + n)
        n += 1
    return np.mean(losses)

def test(encoder, decoder, test_data, loss_func, n_vocab, encoder_device, decoder_device, max_length=128):
    encoder.eval()
    decoder.eval()
    data = random.choice(test_data)

    input_s = torch.tensor([1] + data + [2], device=encoder_device).unsqueeze(0)
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

    with open(str(Path(args.output_dir) / "hyper_param.json"), 'wt') as f:
        json.dump(hyperp, f)

    decoder = transformer_Decoder(n_vocab, d_model, n_head, n_hidden, decoder_nlayers, encoder.bert.get_input_embeddings(), nn.LayerNorm(d_model), dropout=dropout)

    # encoderのBERT内に組み込まれてる BertEmbeddings をdecoderで使うため、GPUへ送る順番は decoder->encoder
    decoder = decoder.to(decoder_device)
    encoder = encoder.to(encoder_device)

    loss_func = nn.CrossEntropyLoss(ignore_index=3)

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
    train_dataloader = get_dataloader(train_dataset, batch_size, pad_index=3, bos_index=1, eos_index=2)

    writer = SummaryWriter(log_dir=args.output_dir)

    t_itr = tqdm.trange(200, leave=False)
    for epoch in t_itr:
        train_loss = train(encoder, decoder, train_dataloader, loss_func, encoder_opt, decoder_opt, n_vocab, encoder_device, decoder_device, writer, epoch)
        t_itr.set_postfix({"ave_loss":train_loss})
        writer.add_scalar('Loss/average', train_loss, epoch)

        input_data, output_data = test(encoder, decoder, train_dataset, loss_func, n_vocab, encoder_device, decoder_device)
        output_text = "{}\n{}\n-> {}\n   {}".format(sp.decode(input_data), input_data, sp.decode(output_data), output_data)
        t_itr.write(output_text)
        writer.add_text("output_text", output_text, epoch)

        torch.save({
                'epoch': true_epoch + epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_opt_state_dict': encoder_opt.state_dict(),
                'decoder_opt_state_dict': decoder_opt.state_dict(),
                'train_loss': train_loss},
            str(Path(args.output_dir) / "{}_epoch{:03d}.pt".format(device_name, epoch)))

    writer.close()
