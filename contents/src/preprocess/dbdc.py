import json
import argparse
from pathlib import Path

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
                    utterances.append(item["utterance"])
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(utterances))

def make_dialog_database(input_dir, output_file):
    utterances = []
    for filename in Path(input_dir).glob("**/*.json"):
        if 'en' not in filename.parts:
            print(filename)
            dialog = []
            with open(filename, 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())

            utt_pair = dict()
            for i, item in enumerate(data["turns"]):
                if item["speaker"] == "S" and i == 0:
                    continue
                elif item["speaker"] == "S":
                    utt_pair["sys_utt"] = item["utterance"]
                elif item["speaker"] == "U":
                    utt_pair["usr_utt"] = item["utterance"]

                if "sys_utt" in utt_pair and "usr_utt" in  utt_pair:
                    dialog.append(utt_pair)
                    utt_pair = dict()
            utterances.append(dialog)

    with open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(utterances, f)

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
    else:
        print('file_type should be "utt_list" or "database"')
