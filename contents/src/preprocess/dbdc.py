import json
import argparse
from pathlib import Path
import unicodedata

import numpy as np

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    return s

def make_uttrence_list(input_dir, output_file):
    utterances = []
    for filename in Path(input_dir).glob("**/*.json"):
        if 'en' not in filename.parts:
            with open(filename, 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())
            for item in data["turns"]:
                if item["speaker"] == "S":
                    ungrammatical = [tmp for tmp in item["annotations"] if tmp["ungrammatical-sentence"] == "X"]
                    ung_rate = len(ungrammatical) / len(item["annotations"])
                else:
                    ung_rate = 0

                if ung_rate < 0.5:
                    utterances.append(normalize_string(item["utterance"]))
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(utterances))

def make_dialog_database(input_dir, output_file):
    utterances = []
    for filename in Path(input_dir).glob("**/*.json"):
        if 'en' not in filename.parts:
            print(filename)
            with open(filename, 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())

            dialog1 = []
            dialog2 = []
            utt_pair1 = dict()
            utt_pair2 = dict()
            for i, item in enumerate(data["turns"]):
                if i == 0:
                    if item["speaker"] == "S":
                        utt_pair2["usr_utt"] = normalize_string(item["utterance"])
                    elif item["speaker"] == "U":
                        utt_pair1["usr_utt"] = normalize_string(item["utterance"])

                    continue

                if item["speaker"] == "S":
                    utt_pair1["sys_utt"] = normalize_string(item["utterance"])
                    utt_pair2["usr_utt"] = normalize_string(item["utterance"])
                elif item["speaker"] == "U":
                    utt_pair1["usr_utt"] = normalize_string(item["utterance"])
                    utt_pair2["sys_utt"] = normalize_string(item["utterance"])

                if "sys_utt" in utt_pair1 and "usr_utt" in utt_pair1:
                    dialog1.append(utt_pair1)
                    utt_pair1 = dict()
                if "sys_utt" in utt_pair2 and "usr_utt" in utt_pair2:
                    dialog2.append(utt_pair2)
                    utt_pair2 = dict()

            utterances.append(dialog1)
            utterances.append(dialog2)

    with open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(utterances, f)

def make_dbdc_data(input_dir):
    score_dict = {"O":1.0, "T":0.5, "X":0.0}
    datas = []
    for filename in Path(input_dir).glob("**/*.json"):
        if 'en' not in filename.parts:
            print(filename)
            with open(filename, 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())

            utterances = []
            for i, item in enumerate(data["turns"]):
                utterances.append(normalize_string(item["utterance"]))
                if item["speaker"] == "S":
                    score = np.mean([score_dict[anno_item["breakdown"]] for anno_item in item["annotations"]])
                    grammar = np.mean([score_dict[anno_item["ungrammatical-sentence"]] for anno_item in item["annotations"]])
                    datas.append({"utterances":utterances.copy(), "score":score, "grammar":grammar})
    return datas

def make_grammar_data(input_dir):
    score_dict = {"O":1.0, "X":0.0}
    datas = []
    for filename in Path(input_dir).glob("**/*.json"):
        if 'en' not in filename.parts:
            print(filename)
            with open(filename, 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())

            for item in data["turns"]:
                if item["speaker"] == "S":
                    grammar = np.mean([score_dict[anno_item["ungrammatical-sentence"]] for anno_item in item["annotations"]])
                    datas.append({"utterance":normalize_string(item["utterance"]), "grammar":grammar})
    return datas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_type")
    parser.add_argument("-i", "--inputdir", required=True)
    parser.add_argument("-o", "--outputfile", required=True)
    args = parser.parse_args()

    if args.file_type == "utt_list":
        make_uttrence_list(args.inputdir, args.outputfile)
    elif args.file_type == "database":
        make_dialog_database(args.inputdir, args.outputfile)
    elif args.file_type == "dbdc_data":
        datas = make_dbdc_data(args.inputdir)
        with open(args.outputfile, 'wt', encoding='utf-8') as f:
            json.dump(datas, f)
    elif args.file_type == "grammar":
        datas = make_grammar_data(args.inputdir)
        with open(args.outputfile, 'wt', encoding='utf-8') as f:
            json.dump(datas, f, indent=2, ensure_ascii=False)
    else:
        print('file_type should be ["utt_list","database","dbdc_data", "grammar"]')
