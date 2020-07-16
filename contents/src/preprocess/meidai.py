import unicodedata
import re
from pathlib import Path
import argparse

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    parser.add_argument("outputdir")
    args = parser.parse_args()

    file_list = [item for item in Path(args.inputdir).glob("data*.txt")]
    file_list.sort()
    for item in file_list:
        lines = []
        with open(item, 'rt') as f:
            for line in f:
                line = normalize_string(line).strip()
                if re.match(r"@|%com:", line) is None:
                    m = re.match(r"(.*?:)(.*)", line)
                    if m:
                        now_talker = m.group(1)
                        line = m.group(2)

                    if re.search(r"\*", line) is None:
                        line = re.sub(r"<.*?>|\(.*?\)|【.*?】", r"", line)
                        if line != "":
                            lines.append((now_talker, line))

        with open(Path(args.outputdir) / item.name, 'wt') as f:
            f.write('\n'.join(["{}:{}".format(item[0], item[1]) for item in lines]) + '\n')
