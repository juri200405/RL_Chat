import argparse
from pathlib import Path
import json
import pickle
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
import tqdm

from config import Config
from vae_check import VAE_tester
import bert_dataloader

class MyModel(torch.nn.Module):
    def __init__(self, n_latent, mid_size):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(n_latent, mid_size)
        self.fc2 = torch.nn.Linear(mid_size, 1)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        o = self.sigmoid(self.fc2(h))
        return o.squeeze()

def get_collate_fn(score_noise):
    def _f(batch):
        tensors = torch.cat([item["tensor"] for item in batch])
        score = torch.tensor([item["score"] for item in batch])
        if score_noise:
            m = torch.distributions.ContinuousBernoulli(torch.zeros_like(score, dtype=torch.float))
            score = torch.abs(score - m.sample())

        return tensors, score
    return _f

def get_dataloader(dataset, batchsize, score_noise=False):
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batchsize,
            sampler=torch.utils.data.sampler.RandomSampler(dataset),
            collate_fn=get_collate_fn(score_noise)
            )
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_data", type=int, default=32768)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--score_noise", action="store_true")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.output_dir)
    print("complete making writer")

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    print("complete loading spm_model")

    config = Config()
    config.load_json(str(Path(args.vae_checkpoint).with_name("hyper_param.json")))
    config.dropout = 0.0
    print("complete loading config")

    device = torch.device("cuda", args.gpu)

    tester = VAE_tester(config, sp, device)
    tester.load_pt(args.vae_checkpoint)
    print("complete loading vae")

    with open(args.input_file, 'rb') as f:
        data_list = pickle.load(f)
    if args.num_data < 0 or args.num_data > len(data_list):
        sampled_data = random.sample(data_list, len(data_list))
    else:
        sampled_data = random.sample(data_list, args.num_data)
    true_dataloader = bert_dataloader.get_dataloader(sampled_data, 2048, pad_index=3, bos_index=1, eos_index=2, fix_max_len=128, fix_len=128)
    true_datas = []
    with torch.no_grad():
        for inputs, mask in tqdm.tqdm(true_dataloader):
            true_datas += [{"tensor":item, "score":1.0} for item in tester.encoder(inputs.to(device), attention_mask=mask.to(device)).cpu().split(1)]

    false_data_size = len(true_datas)
    false_datas = [{"tensor": item, "score": 0.0} for item in tqdm.tqdm(torch.randn(false_data_size, config.n_latent).split(1))]

    datas = true_datas + false_datas
    val_size = int(len(datas) * 0.1)
    train_datas, val_datas = torch.utils.data.random_split(datas, [len(datas) - val_size, val_size])
    train_dataloader = get_dataloader(train_datas, 128, args.score_noise)
    val_dataloader = get_dataloader(val_datas, 128, False)
    print("complete data preparation")

    model = MyModel(config.n_latent, 2048).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()

    epoch_itr = tqdm.trange(args.num_epoch)
    for epoch in epoch_itr:
        t_itr = tqdm.tqdm(train_dataloader)
        model.train()
        for i, batch in enumerate(t_itr):
            tensors, score = batch
            tensors = tensors.to(device)
            score = score.to(device)

            out = model(tensors)
            loss = loss_func(out, score)

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), i)
            t_itr.set_postfix({"loss":loss.item()})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                tensors, score = batch
                tensors = tensors.to(device)
                score = score.to(device)

                out = model(tensors)
                loss = loss_func(out, score)
                val_loss += loss.item()
        val_loss = val_loss / val_size
        writer.add_scalar("val_loss", val_loss, epoch)
        epoch_itr.set_postfix({"val_loss": val_loss})

        torch.save(model.state_dict(), str(Path(args.output_dir)/"epoch{:03d}.pt".format(epoch)))
