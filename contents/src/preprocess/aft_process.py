import re
import argparse

def line_process(s):
    return s

def process(input_file):
    processed_lines = []

    with open(input_file, "rt", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # unicodeの漢字ブロックの終わり以降のコードを含む行の削除
            m = re.search("[\uA000-\uF8FF]", line)
            if m is not None:
                continue

            # unicodeの漢字の部首が出現するまでのコードを含む行の削除
            m = re.search("[\u00A0-\u2E52]", line)
            if m is not None:
                continue

            # 各行内での処理
            line = line_process(line)
            processed_lines.append(line)

    return processed_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    args = parser.parse_args()

    lines = process(args.input_file)

    with open(args.output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(lines))
