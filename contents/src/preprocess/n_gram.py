import pickle
import argparse
from pathlib import Path

import tqdm

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

    n_grams = dict()
    total = 0
    for sentence in tqdm.tqdm(datas):
        n_gram = [tuple(sentence[i:i+args.n_gram]) for i in range(len(sentence)+1 - args.n_gram)]
        for item in n_gram:
            total += 1
            if item in n_grams:
                n_grams[item] += 1
            else:
                n_grams[item] = 1

    with open(args.outputfile, "wb") as f:
        pickle.dump((n_grams, total), f)
