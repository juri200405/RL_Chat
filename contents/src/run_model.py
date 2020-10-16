import argparse
from pathlib import Path
import random
import json
from itertools import chain

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

from bert_data import txt_to_idlist
from bert_dataloader import get_dataloader
from encoder_decoder import Bert_Encoder_vae, transformer_Decoder, Transformer_Embedding, transformer_Encoder
from config import Config
from losses import VaeLoss


def train(encoder, decoder, train_dataloader, loss_func, encoder_opt, decoder_opt, config, writer, epoch, output_dir=Path("")):
    encoder.train()
    decoder.train()
    losses = []
    train_itr = tqdm.tqdm(train_dataloader, leave=False, ncols=150)
    n = 0
    for sentence, inp_padding_mask, tgt_padding_mask in train_itr:
    # for sentence, _ in train_itr:
        inputs = sentence[:,:]
        tgt = sentence[:,:]
        label = sentence[:,1:]

        inputs = inputs.to(config.encoder_device)
        inp_padding_mask = inp_padding_mask.to(config.encoder_device)
        # # embedding がencoderのdeviceにあるため、tgtはencoder_deviceに送る
        # tgt = tgt.to(config.encoder_device)
        tgt = tgt.to(config.decoder_device)
        label = label.to(config.decoder_device)
        tgt_padding_mask = tgt_padding_mask.to(config.decoder_device)

        if config.model_type == "bert":
            m, memory = encoder(inputs, attention_mask=inp_padding_mask)
        else:
            m, memory = encoder(inputs, attention_mask=tgt_padding_mask)
        memory = memory.to(config.decoder_device)
        out = decoder(tgt, memory, tgt_padding_mask=tgt_padding_mask)
        # make_dot(out).render(str(output_dir + "graph"))

        out = out[:-1].contiguous().view(-1, out.shape[-1])
        label = label.transpose(0,1).contiguous().view(-1)
        
        loss, cross_entropy, kl_loss, kl_weight = loss_func(out, label, m, epoch * len(train_itr) + n)

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
        encoder_opt.step()
        decoder_opt.step()

        losses.append(loss.item())
        train_itr.set_postfix({"loss":loss.item(), "weight":kl_weight, "kl_loss":kl_loss.item()})
        writer.add_scalar('Loss/each',loss.item(), epoch * len(train_itr) + n)
        writer.add_scalar('Detail_Loss/cross_entropy', cross_entropy.item(), epoch * len(train_itr) + n)
        writer.add_scalar('Detail_Loss/kl_loss', kl_loss.item(), epoch * len(train_itr) + n)
        writer.add_scalar('Detail_Loss/kl_weight', kl_weight, epoch * len(train_itr) + n)

        n += 1
    return np.mean(losses)

def test(encoder, decoder, test_data, loss_func, config):
    encoder.eval()
    decoder.eval()
    data = random.choice(test_data)

    input_s = torch.tensor([1] + data + [2], device=config.encoder_device).unsqueeze(0)
    _, memory = encoder(input_s)
    memory = memory.to(config.decoder_device)

    tgt = torch.full((1, config.max_len), 0, dtype=torch.long, device=config.encoder_device)
    tgt_key_padding_mask = torch.full((1, config.max_len), True, dtype=torch.bool, device=config.decoder_device)
    t_word = 1 # <s>
    ids = []
    for t in range(config.max_len):
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

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    model_config = Config()
    model_config.load_json(args.hyper_param)
    model_config.n_vocab = len(sp)

    print(args.output_dir)
    print("encoder = {}, decoder = {}".format(model_config.encoder_device, model_config.decoder_device))

    if model_config.model_type == "bert":
        encoder = Bert_Encoder_vae(args.bert_path)
        model_config.d_model = encoder.bert.config.hidden_size
        embedding_model = encoder.bert.get_input_embeddings()
    elif model_config.model_type == "transformer":
        # embedding_model = Transformer_Embedding(model_config)
        # encoder = transformer_Encoder(model_config, embedding_model, nn.LayerNorm(model_config.d_model))
        encoder = transformer_Encoder(model_config, Transformer_Embedding(model_config), nn.LayerNorm(model_config.d_model))
    else:
        print("model_type missmatch")
        exit()

    train_dataset = txt_to_idlist(sp, args.input_file, 3)
    train_dataloader = get_dataloader(train_dataset, model_config.batch_size, pad_index=3, bos_index=1, eos_index=2)

    loss_func = VaeLoss(nn.CrossEntropyLoss(ignore_index=3, reduction='sum'), model_config, len(train_dataloader)).forward
    model_config.save_json(str(Path(args.output_dir) / "hyper_param.json"))

    # decoder = transformer_Decoder(model_config, embedding_model, nn.LayerNorm(model_config.d_model))
    decoder = transformer_Decoder(model_config, Transformer_Embedding(model_config), nn.LayerNorm(model_config.d_model))

    # encoderのBERT内に組み込まれてる BertEmbeddings をdecoderで使うため、GPUへ送る順番は decoder->encoder
    decoder = decoder.to(model_config.decoder_device)
    encoder = encoder.to(model_config.encoder_device)


    if model_config.optim_type == "Adam":
        encoder_opt = optim.Adam(encoder.parameters())
        decoder_opt = optim.Adam(decoder.parameters())
    elif model_config.optim_type == "RAdam":
        encoder_opt = torch_optimizer.RAdam(encoder.parameters())
        decoder_opt = torch_optimizer.RAdam(decoder.parameters())
    elif model_config.optim_type == "Yogi":
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

    writer = SummaryWriter(log_dir=args.output_dir)
    with open(str(Path(args.output_dir) / "out_text.csv"), 'w', encoding='utf-8') as out:
        out.write("input_text,reconstract_text")

    t_itr = tqdm.trange(model_config.num_epoch, leave=False, ncols=150)
    for epoch in t_itr:
        train_loss = train(encoder, decoder, train_dataloader, loss_func, encoder_opt, decoder_opt, model_config, writer, epoch, args.output_dir)
        t_itr.set_postfix({"ave_loss":train_loss})
        writer.add_scalar('Loss/average', train_loss, epoch)

        input_data, output_data = test(encoder, decoder, train_dataset, loss_func, model_config)
        output_text = "{}\n{}\n-> {}\n   {}".format(sp.decode(input_data), input_data, sp.decode(output_data), output_data)
        t_itr.write(output_text)
        writer.add_text("output_text", output_text, epoch)

        with open(str(Path(args.output_dir) / "out_text.csv"), 'a', encoding='utf-8') as out:
            out.write("\n{},{}".format(sp.decode(input_data), sp.decode(output_data)))

        torch.save({
                'epoch': true_epoch + epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_opt_state_dict': encoder_opt.state_dict(),
                'decoder_opt_state_dict': decoder_opt.state_dict(),
                'train_loss': train_loss},
            str(Path(args.output_dir) / "epoch{:03d}.pt".format(epoch)))

    writer.close()
