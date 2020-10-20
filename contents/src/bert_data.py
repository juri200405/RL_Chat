from typing import List
import pickle
import argparse

import sentencepiece as spm

def txt_to_idlist(spm_model, input_file: str, sample_num=0) -> List[List[int]]:
    sp = spm_model

    numericalized_lines = []
    with open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if sample_num == 0:
                numericalized_lines.append(sp.encode(line))
            else:
                for _ in range(sample_num):
                    numericalized_lines.append(sp.encode(line, enable_sampling=True, nbest_size=-1, alpha=0.1))

    return numericalized_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    train_dataset = txt_to_idlist(sp, args.input_file, 3)

    with open(args.output_file, 'wb') as f:
        pickle.dump(train_dataset, f)
