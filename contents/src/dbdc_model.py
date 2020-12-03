import argparse
from pathlib import Path
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import tqdm
import numpy as np

from encoder_decoder import transformer_Encoder, Transformer_Embedding
from config import Config

class DBDC(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, ffl_hidden):
        super(DBDC, self).__init__()
        self.hidden_size = hidden_size

        self.gru = torch.nn.GRU(input_size=latent_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, ffl_hidden)
        self.fc2 = torch.nn.Linear(ffl_hidden, 1)

        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, conv_len):
        hidden = torch.zeros(1, x.shape[0], self.hidden_size, device=x.device)
        outputs, _ = self.gru(x, hidden)

        idx = (conv_len-1).view(-1, 1).expand(outputs.shape[0], outputs.shape[2]).unsqueeze(1)
        out = outputs.gather(1, idx).squeeze()

        out = self.fc1(out)
        out = self.fc2(self.relu(out))
        return self.sigmoid(out).squeeze()

def encode(x, mask, batchsize, encoder):
    with torch.no_grad():
        x = encoder(x, attention_mask=mask)
        x = x.view(batchsize, -1, x.shape[1])
    return x

def process_data(pair, sp, max_len):
    data = pair["utterances"]
    data = sp.encode(data)

    tensors = [torch.LongTensor([1] + item + [2]) for item in data]

    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=3)
    pad = torch.full((padded_tensor.shape[0], max_len-padded_tensor.shape[1]), 3, dtype=torch.long)
    padded_tensor = torch.cat((padded_tensor, pad), dim=1)

    ids_len = [item.shape[0] for item in tensors]
    padding_mask = torch.tensor([[False]*i + [True]*(padded_tensor.shape[1]-i) for i in ids_len])

    pair["utterances"] = padded_tensor
    pair["mask"] = padding_mask

    return pair

def get_collate_fn():
    def _f(batch):
        scores = torch.tensor([item["score"] for item in batch])
        padded_tensors = [item["utterances"] for item in batch]
        padding_masks = [item["mask"] for item in batch]

        conv_len = [item.shape[0] for item in padded_tensors]
        max_conv_len = max(conv_len)

        # padded_tensors = torch.cat([torch.cat((item, torch.full((max_conv_len -item.shape[0], item.shape[1]), 3, dtype=torch.long)), dim=0) for item in padded_tensors], dim=0)
        padded_tensors = torch.cat([torch.cat((item, torch.tensor([[1]+[2]+[3]*(item.shape[1]-2)]*(max_conv_len-item.shape[0]), dtype=torch.long)), dim=0) for item in padded_tensors], dim=0)
        # padding_masks = torch.cat([torch.cat((item, torch.full((max_conv_len -item.shape[0], item.shape[1]), True, dtype=torch.bool)), dim=0) for item in padding_masks], dim=0)
        padding_masks = torch.cat([torch.cat((item, torch.tensor([[False]*2+[True]*(item.shape[1]-2)]*(max_conv_len-item.shape[0]), dtype=torch.bool)), dim=0) for item in padding_masks], dim=0)

        return {"input": padded_tensors, "mask": padding_masks, "conv_len": torch.tensor(conv_len), "score":scores}

    return _f

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gru_hidden", type=int, default=128)
    parser.add_argument("--ff_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=100)
    args = parser.parse_args()

    hyper_param = Path(args.vae_checkpoint).parent / "hyper_param.json"

    config = Config()
    config.load_json(str(hyper_param))
    device = torch.device("cuda:{}".format(args.gpu))

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    writer = SummaryWriter(log_dir=args.output_dir)
    with open(str(Path(args.output_dir)/"dbdc_model_param.json"), "wt", encoding="utf-8") as f:
        json.dump(vars(args), f)

    encoder = transformer_Encoder(config, Transformer_Embedding(config), torch.nn.LayerNorm(config.d_model))

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()

    with open(args.data_file, "rt", encoding="utf-8") as f:
        datas = json.load(f)
    dataset = [process_data(item, sp, config.max_len) for item in datas]
    val_size = int(len(dataset)*0.1)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.RandomSampler(train_dataset),
            collate_fn=get_collate_fn(),
            num_workers=2,
            pin_memory=True
            )
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False,
            collate_fn=get_collate_fn(),
            num_workers=2,
            pin_memory=True
            )

    model = DBDC(config.n_latent, args.gru_hidden, args.ff_hidden)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

    for i in tqdm.tqdm(range(args.num_epoch)):
        for n, item in enumerate(tqdm.tqdm(train_dataloader)):
            model.train()
            x = encode(item["input"].to(device), item["mask"].to(device), item["conv_len"].shape[0], encoder)
            out = model(x, item["conv_len"].to(device))
            loss = loss_func(out, item["score"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar("train/loss", loss.item(), i*len(train_dataloader) + n)

        val_loss = []
        for item in tqdm.tqdm(val_dataloader):
            model.eval()
            x = encode(item["input"].to(device), item["mask"].to(device), item["conv_len"].shape[0], encoder)
            out = model(x, item["conv_len"].to(device))
            loss = loss_func(out, item["score"].to(device))
            val_loss.append(loss.item())
        writer.add_scalar("val/loss", np.mean(val_loss), i)

        torch.save({
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict()
            }, str(Path(args.output_dir)/"epoch{:03d}.pt".format(i)))

    writer.add_scalar("val/loss", np.mean(val_loss), i)
    writer.close()
