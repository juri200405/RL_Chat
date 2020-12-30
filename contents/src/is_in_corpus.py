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

def get_collate_fn():
    def _f(batch):
        tensors = torch.cat([item["tensor"] for item in batch])
        score = torch.tensor([item["score"] for item in batch])

        return tensors, score
    return _f

def get_dataloader(dataset, batchsize):
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batchsize,
            sampler=torch.utils.data.sampler.RandomSampler(dataset),
            collate_fn=get_collate_fn()
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
    sampled_data = random.sample(data_list, args.num_data)
    true_dataloader = bert_dataloader.get_dataloader(sampled_data, 2048, pad_index=3, bos_index=1, eos_index=2, fix_max_len=128, fix_len=128)
    true_datas = []
    with torch.no_grad():
        for inputs, mask in tqdm.tqdm(true_dataloader):
            true_datas += [{"tensor":item, "score":1.0} for item in tester.encoder(inputs.to(device), attention_mask=mask.to(device)).cpu().split(1)]

    false_data_size = int(len(true_datas) * 1.2)
    false_datas = [{"tensor": item, "score": 0.0} for item in tqdm.tqdm(torch.randn(false_data_size, config.n_latent).split(1))]
    datas = true_datas + false_datas
    dataloader = get_dataloader(datas, 128)
    print("complete data preparation")

    model = MyModel(config.n_latent, 2048).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()

    t_itr = tqdm.tqdm(dataloader)
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

    torch.save(model.state_dict(), str(Path(args.output_dir)/"checkpoint.pt"))
