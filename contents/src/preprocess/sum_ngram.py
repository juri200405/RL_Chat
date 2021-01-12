import pickle
import argparse
from pathlib import Path
import math
import random

import tqdm

import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", default=None)
    parser.add_argument("--ngram_list", default=None)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--n_gram", type=int, default=3)
    args = parser.parse_args()

    output_path = Path(args.outputdir)
    if not output_path.exists():
        output_path.mkdir()

    if args.inputfile is not None and args.ngram_list is None:
        input_file = Path(args.inputfile)
        if input_file.suffix == ".pkl":
            with open(str(input_file), "rb") as f:
                datas = pickle.load(f)
                ngram_list = [[tuple(sentence[i:i+args.n_gram]) for i in range(len(sentence)+1 - args.n_gram)] for sentence in tqdm.tqdm(datas) if len(sentene)>0]
                del datas
        elif input_file.suffix == ".txt":
            with open(str(input_file), "rt", encoding="utf-8") as f:
                ngram_list = [[sentence[i:i+args.n_gram] for i in range(len(sentence)+1 - args.n_gram)] for sentence in tqdm.tqdm(f) if len(sentence)>0]
        else:
            print("input_file must be .pkl file")
            exit()

        ngram2id = {tmp:i for i, tmp in enumerate({item for sentence in ngram_list for item in sentence})}
        num_ngram = len(ngram2id)
        with open(str(output_path/"ngram2id.pkl"), "wb") as f:
            pickle.dump(ngram2id, f)

        ngram_list = [[ngram2id[item] for item in sentence] for sentence in tqdm.tqdm(ngram_list) if len(sentence) > 0]
        with open(str(output_path/"ngram_list.pkl"), "wb") as f:
            pickle.dump((ngram_list, num_ngram), f)
        del ngram2id
    elif args.inputfile is None and args.ngram_list is not None:
        with open(args.ngram_list, "rb") as f:
            ngram_list, num_ngram = pickle.load(f)
    else:
        print("incorrect options")
        exit()

    sum_ngram = torch.zeros(1, num_ngram)
    for sentence in tqdm.tqdm(ngram_list):
        sum_ngram = sum_ngram + torch.nn.functional.one_hot(torch.tensor(sentence), num_classes=num_ngram).sum(dim=0, keepdim=True)
    torch.save(sum_ngram, str(output_path/"sum_ngram.pt"))
    cos = torch.nn.CosineSimilarity()
    cos_list = [cos(sum_ngram, torch.nn.functional.one_hot(torch.tensor(sentence), num_classes=num_ngram).sum(dim=0, keepdim=True)) for sentence in tqdm.tqdm(ngram_list)]
    print(min(cos_list))
