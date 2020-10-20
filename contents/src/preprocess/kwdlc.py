import argparse
from pathlib import Path
import re
import unicodedata

import tqdm

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    return s

def file2list(filename):
    lines = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = re.match(r"# ", line)
            if m is None and len(line) > 1:
                lines.append(normalize_string(line))
    return lines

def process_dir(input_dir):
    lines = []
    t_itr = tqdm.tqdm(Path(input_dir).glob("**/*.org"))
    for filename in t_itr:
        lines += file2list(filename)
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    args = parser.parse_args()

    lines = process_dir(args.input_dir)
    with open(args.output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(lines))
