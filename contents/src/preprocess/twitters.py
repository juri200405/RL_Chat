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
    rm_lines = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = normalize_string(line)
            line = line.strip()
            line = re.sub(r"\s", "", line)

            # 写真urlの削除
            line = re.sub(r"pic\.[./a-zA-Z0-9_]+", "", line)

            # メンションの削除
            line = re.sub(r"@[a-zA-Z0-9_]+", "", line)

            m = re.fullmatch(r"{low}|<\|endoftext\|>", line)
            rm = re.search(r"[^\w!?。、.,～〜「」『』()・/]", line)
            if m is None and rm is None and len(line) > 1:
                lines.append(line)
            elif rm is not None and m is None:
                rm_lines.append(line)
    return lines, rm_lines

def process_dir(input_dir):
    lines = []
    rm_lines = []
    t_itr = tqdm.tqdm(Path(input_dir).glob("**/*.txt"))
    for filename in t_itr:
        t_itr.write(str(filename))
        line, rm_line = file2list(filename)
        lines += line
        rm_lines += rm_line
    return lines, rm_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-o2", "--output_file2", required=True)
    args = parser.parse_args()

    lines, rm_lines = process_dir(args.input_dir)
    with open(args.output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    with open(args.output_file2, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(rm_lines))
