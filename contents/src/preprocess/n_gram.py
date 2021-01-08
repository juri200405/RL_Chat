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
    n_grams = {item for sentence in ngram_list for item in sentence}
    ngram2id = {item:i for i, item in enumerate(n_grams)}

    ngram_vectors = torch.zeros(len(ngram_list), len(n_grams), dtype=torch.float)
    for i, sentence in enumerate(tqdm.tqdm(ngram_list)):
        for item in sentence:
            ngram_vectors[i, ngram2id[item]] += 1.0

    with open(args.outputfile, "wb") as f:
        pickle.dump((ngram_vectors, ngram2id), f)
