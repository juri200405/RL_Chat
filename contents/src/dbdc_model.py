import arguparse

import torch
import pytorch_lightning as pl

from encoder_decoder import MMD_VAE
from config import Config

class DBDC(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, ffl_hidden, dropout):
        self.hidden_size = hidden_size

        self.gru = torch.nn.GRU(input_size=latent_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.fc1 = torch.nn.Linear(hidden_size, ffl_hidden)
        self.fc2 = torch.nn.Linear(ffl_hidden, 1)

        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros(1, x.shape[0], self.hidden_size)
        _, hidden = self.gru(x)
        out = self.fc1(hidden.squeeze(0))
        out = self.fc2(self.relu(out))
        return self.sigmoid(out)

if __name__ == "__main__":
    parser.argparse.ArgumentParser()
    parser.add_argument("--hyper_param", required=True)
    parser.add_argument("--mmdvae_checkpoint", required=True)
    parser = MMD_VAE.add_argparse_args(parser)
    args = parser.parse_args()

    config = Config()
    config.load_json(args.hyper_param)
    model = MMD_VAE.load_from_checkpoint(args.mmdvae_checkpoint, config, args)

    encoder = model.encoder
    decoder = model.decoder

    loss_func = torch.nn.MSELoss()
