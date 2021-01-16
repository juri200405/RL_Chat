import random

import torch
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import tqdm

from config import Config
from bert_dataloader import get_dataloader
from encoder_decoder import Transformer_Embedding, transformer_Encoder

if __name__ == "__main__":
    with open("data/tomioka/no_kwdlc.txt", "rt", encoding="utf-8") as f:
        datas = [item for item in f if len(item)>0]

    sp = spm.SentencePieceProcessor(model_file="data/tomioka/spm_model/no_kwdlc.model")
    config = Config()
    config.load_json("data/tomioka/output/transformer_vae/2020_12_26/1/hyper_param.json")
    config.dropout = 0.0

    device = torch.device("cuda", 0)

    encoder = transformer_Encoder(
            config,
            Transformer_Embedding(config),
            torch.nn.LayerNorm(config.d_model)
            )
    encoder.to(device)

    encoder.load_state_dict(
            torch.load(
                "data/tomioka/output/transformer_vae/2020_12_26/1/epoch001.pt",
                map_location=device
                )["encoder_state_dict"]
            )

    data_loader = get_dataloader(
            sp.encode(datas),
            16384,
            pad_index=3,
            bos_index=1,
            eos_index=2,
            fix_max_len=config.max_len,
            fix_len=config.max_len,
            shuffle=False
            )

    for sentence, mask in tqdm.tqdm(data_loader):
        sentence = sentence.to(device)
        mask = mask.to(device)

        with open("data/output/z/tensors.tsv", "at", encoding="utf-8") as f:
            f.write("\n".join(["\t".join([str(item) for item in sentence]) for sentence in encoder(sentence, attention_mask=mask).cpu().tolist()]))
