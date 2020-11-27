import arguparse

import torch

from encoder_decoder import transformer_Encoder, Transformer_Embedding
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

def DBDC_prepare(encoder, datas):
    processed = []

if __name__ == "__main__":
    parser.argparse.ArgumentParser()
    parser.add_argument("--hyper_param", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    args = parser.parse_args()

    config = Config()
    config.load_json(args.hyper_param)

    encoder = transformer_Encoder(config, Transformer_Embedding(config), nn.LayerNorm(config.d_model))

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
    self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    loss_func = torch.nn.MSELoss()
