from typing import List
import pickle
import argparse

import sentencepiece as spm

def txt_to_idlist(spm_model, input_file: str, sample_num=0, max_len=None) -> List[List[int]]:
    sp = spm_model

    numericalized_lines = []
    with open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if sample_num == 0:
                tokens = sp.encode(line)
                if max_len is None or len(tokens) <= max_len:
                    numericalized_lines.append(tokens)
            else:
                for _ in range(sample_num):
                    tokens = sp.encode(line, enable_sampling=True, nbest_size=-1, alpha=0.1)
                    if max_len is None or len(tokens) <= max_len:
                        numericalized_lines.append(tokens)

    return numericalized_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-n", "--n_sample", default=3, type=int)
    parser.add_argument("-l", "--max_len", default=0, type=int)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    if args.max_len == 0:
        train_dataset = txt_to_idlist(sp, args.input_file, args.n_sample)
    else:
        train_dataset = txt_to_idlist(sp, args.input_file, args.n_sample, max_len=args.max_len)

    with open(args.output_file, 'wb') as f:
        pickle.dump(train_dataset, f)
