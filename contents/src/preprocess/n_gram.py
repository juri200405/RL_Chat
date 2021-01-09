import pickle
import argparse
from pathlib import Path

import tqdm

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile", required=True)
    parser.add_argument("--n_gram", type=int, default=3)
    args = parser.parse_args()

    input_file = Path(args.inputfile)
    if input_file.suffix == ".pkl":
        with open(str(input_file), "rb") as f:
            datas = pickle.load(f)
    else:
        print("input_file must be .pkl file")
        exit()

    ngram_list = [[tuple(sentence[i:i+args.n_gram]) for i in range(len(sentence)+1 - args.n_gram)] for sentence in tqdm.tqdm(datas)]
    del datas
    ngram2id = {tmp:i for i, tmp in enumerate({item for sentence in ngram_list for item in sentence})}
    num_ngram = len(ngram2id)

    ngram_list = [[ngram2id[item] for item in sentence] for sentence in tqdm.tqdm(ngram_list) if len(sentence) > 0]
    # ngram_vectors = torch.cat([torch.nn.functional.one_hot(torch.tensor(sentence), num_classes=num_ngram).sum(dim=0, keepdim=True) for sentence in ngram_list], dim=0).float()

    with open(args.outputfile, "wb") as f:
        # pickle.dump((ngram_vectors, ngram2id), f)
        pickle.dump((ngram_list, ngram2id, num_ngram), f)
