import argparse
from pathlib import Path
import json

import torch
import sentencepiece as spm
import tqdm

from encoder_decoder import transformer_Encoder, Transformer_Embedding
from config import Config

class DBDC(torch.nn.Module):
    def __init__(self, encoder, latent_size, hidden_size, ffl_hidden):
        super(DBDC, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = encoder

        self.gru = torch.nn.GRU(input_size=latent_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, ffl_hidden)
        self.fc2 = torch.nn.Linear(ffl_hidden, 1)

        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask, conv_len):
        with torch.no_grad():
            x = self.encoder(x, attention_mask=mask)
            print(x)
            x = x.view(conv_len.shape[0], -1, x.shape[1])

        hidden = torch.zeros(1, x.shape[0], self.hidden_size)
        outputs, _ = self.gru(x, hidden)

        idx = (conv_len-1).view(-1, 1).expand(outputs.shape[0], outputs.shape[2]).unsqueeze(1)
        out = outputs.gather(1, idx).squeeze()

        out = self.fc1(out)
        out = self.fc2(self.relu(out))
        return self.sigmoid(out).squeeze()

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

        padded_tensors = torch.cat([torch.cat((item, torch.full((max_conv_len -item.shape[0], item.shape[1]), 3, dtype=torch.long)), dim=0) for item in padded_tensors], dim=0)
        padding_masks = torch.cat([torch.cat((item, torch.full((max_conv_len -item.shape[0], item.shape[1]), True, dtype=torch.bool)), dim=0) for item in padding_masks], dim=0)

        return {"input": padded_tensors, "mask": padding_masks, "conv_len": torch.tensor(conv_len), "scores":scores}

    return _f

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()

    hyper_param = Path(args.vae_checkpoint).parent / "hyper_param.json"

    config = Config()
    config.load_json(str(hyper_param))

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    encoder = transformer_Encoder(config, Transformer_Embedding(config), torch.nn.LayerNorm(config.d_model))

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    with open(args.data_file, "rt", encoding="utf-8") as f:
        datas = json.load(f)
    dataset = [process_data(item, sp, config.max_len) for item in datas]
    val_size = int(len(dataset)*0.1)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64,
            sampler=torch.utils.data.sampler.RandomSampler(dataset),
            collate_fn=get_collate_fn(),
            num_workers=2, pin_memory=True
            )
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64,
            shuffle=False,
            collate_fn=get_collate_fn(),
            num_workers=2, pin_memory=True
            )

    model = DBDC(encoder, config.n_latent, 128, 128)
    model.train()
    for item in train_dataloader:
        out = model(item["input"], item["mask"], item["conv_len"])
        break

    loss_func = torch.nn.MSELoss()
