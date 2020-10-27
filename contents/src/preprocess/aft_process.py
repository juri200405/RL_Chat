import re
import argparse

import tqdm


def line_process(s, kaomoji_pattern):
    # 行頭が各種記号だったら削除
    s = re.sub(r"^[/)、。!?.,」』_-]+", "", s)

    # ()の中身が空白 or ! or ? or . or , or /  のものをスペースに
    s = re.sub(r"\([\s!?.,/_]*\)", " ", s)

    # 対応するかっこのないカッコを削除
    while re.search(r"\([^(]*?\)", s) is not None:
        s = re.sub(r"\(([^(]*?)\)", "\u0B67\\1\u0B68", s)
    s = re.sub(r"[()]", " ", s)

    tmp_char = ord("\u1500")
    substr_dict = {}
    while "\u0B67" in s:
        for item in re.finditer("\u0B67([^\u0B67]*?)\u0B68", s):
            substr_dict[chr(tmp_char)] = item.group(1)
            s = re.sub(re.escape("\u0B67{}\u0B68".format(item.group(1))), chr(tmp_char), s)
            tmp_char += 1

    tmp_char -= 1
    while tmp_char >= ord("\u1500"):
        if len(s) == 1:
            s = re.sub(re.escape(chr(tmp_char)), substr_dict[chr(tmp_char)], s)
        elif len(substr_dict[chr(tmp_char)]) == 1 and substr_dict[chr(tmp_char)] in substr_dict:
            s = re.sub(re.escape(chr(tmp_char)), substr_dict[chr(tmp_char)], s)
        else:
            s = re.sub(re.escape(chr(tmp_char)), "(" + substr_dict[chr(tmp_char)] + ")", s)
        tmp_char -= 1

    s = re.sub("\u0B67", "(", s)
    s = re.sub("\u0B68", ")", s)

    # 顔文字をスペースに変換
    s = re.sub(kaomoji_pattern, " ", s)

    # 連続する空白文字をひとつのスペースに
    s = re.sub(r"\s+", " ", s)
    return s

def is_includeZH(s):
    # SJISに変換して文字数が減れば簡体字がある
    # https://qiita.com/ry_2718/items/47c21792d7bbd3fe33b9
    return len(set(s) - set(s.encode('sjis','ignore').decode('sjis'))) > 0

def process(input_list, kaomoji_pattern):
    processed_lines =[] 

    t_itr = tqdm.tqdm(input_list, ncols=100)
    for line in t_itr:
        line = line.strip()
        # unicodeの漢字ブロック以外のコードを含む行の削除
        m = re.search("[\u00A0-\u2E52\uA000-\uF8FF\uFB00-\uFDFF\uFE70-\uFEF0\uFFA0-\uFFEF]", line)
        if m is not None:
            continue

        # ひらがなっぽい別の文字を削除
        m = re.search(r"[ㄘ]", line)
        if m is not None:
            continue

        # 簡体字を含む行を削除
        if is_includeZH(line):
            continue

        # 各行内での処理
        while True:
            inp_line = line
            line = line_process(inp_line, kaomoji_pattern)
            line = line.strip()
            if inp_line == line or len(line) < 1:
                break

        # 英数字だけで構成されている行の削除
        m = re.fullmatch(r"[0-9a-zA-Z_.,!?()]+", line)
        if m is not None:
            continue

        # 空行の削除
        if len(line) < 1:
            continue

        processed_lines.append(line)

    return processed_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-k", "--kaomoji_file", required=True)
    args = parser.parse_args()

    with open(args.kaomoji_file, "rt", encoding="utf-8") as f:
        kaomoji_list = [item.strip() for item in f]
    kaomoji = "|".join([re.escape(item) for item in kaomoji_list])
    kaomoji = re.compile(kaomoji)

    with open(args.input_file, "rt", encoding="utf-8") as f:
        inp_lines = [item for item in f]

    lines = process(inp_lines, kaomoji)
    lines.sort()

    with open(args.output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(lines))
