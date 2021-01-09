import pickle
import argparse
from pathlib import Path

import tqdm

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputdir", required=True)
    args = parser.parse_args()

    input_file = Path(args.inputfile)
    if input_file.suffix == ".pkl":
        with open(str(input_file), "rb") as f:
            ngram_list, _, num_ngram = pickle.load(f)
    else:
        print("input_file must be .pkl file")
        exit()

    output_path = Path(args.outputdir)
    if not output_path.exists():
        output_path.mkdir()

    num_loop = len(ngram_list) // 100
    for i in tqdm.trange(num_loop):
        ngram_vectors = torch.zeros(100, num_ngram, dtype=torch.float)
        for j in range(100):
            sentence = ngram_list.pop()
            for item in sentence:
                ngram_vectors[j, item] += 1.0
        with open(str(output_path/"{:06d}.pkl".format(i)), "wb") as f:
            pickle.dump(ngram_vectors, f)
        del ngram_vectors

    i = i+1
    last_num = len(ngram_list)
    ngram_vectors = torch.zeros(last_num, num_ngram, dtype=torch.float)
    for j in range(last_num):
        sentence = ngram_list.pop()
        for item in sentence:
            ngram_vectors[j, item] += 1.0
    with open(str(output_path/"{:06d}.pkl".format(i)), "wb") as f:
        pickle.dump(ngram_vectors, f)
