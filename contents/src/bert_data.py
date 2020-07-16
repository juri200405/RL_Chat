from typing import List

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
    print(txt_to_idlist("data/tomioka/spm/dbdc_and_meidai_8000.model", "test.txt", 3))
