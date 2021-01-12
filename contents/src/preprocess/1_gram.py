import pickle
import argparse
from pathlib import Path
import math
import random

import tqdm

import torch
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--sp_num", type=int, default=1000)
    args = parser.parse_args()

    output_path = Path(args.outputdir)
    if not output_path.exists():
        output_path.mkdir()

    with open(args.inputfile, "rb") as f:
        ngram_list = pickle.load(f)
    ngram_list = [item for item in ngram_list if len(item)>0]

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    num_ngram = len(sp)

    random.shuffle(ngram_list)
    sp = args.sp_num
    sp_num = math.ceil(len(ngram_list) / sp)
    for i in tqdm.trange(sp):
        ngram_vectors = torch.cat([torch.nn.functional.one_hot(torch.tensor(sentence), num_classes=num_ngram).sum(dim=0, keepdim=True) for sentence in tqdm.tqdm(ngram_list[i*sp_num : (i+1)*sp_num])], dim=0).float()
        torch.save(ngram_vectors, str(output_path/"{:05d}.pt".format(i)))
        del ngram_vectors
