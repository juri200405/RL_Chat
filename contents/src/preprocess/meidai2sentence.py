from pathlib import Path
import re
import argparse

def make_file(in_path, out_path):
    s = ""
    for filename in in_path.glob("data*.txt"):
        with open(filename, 'rt') as f:
            for line in f:
                m = re.match(r".+:(.+)", line)
                if m is None:
                    print(line)
                s += m.group(1)
                s += '\n'
    
    #s = re.sub(r"([。?])", r"\1\n", s)
    with open(str(out_path / "meidai.txt"), 'wt', encoding='utf-8') as f:
        f.write(s)

def make_spm_model(in_file, out_name, vocab_size):
    spm.SentencePieceTrainer.train('--input={} --model_prefix={} --vocab_size={}'.format(in_file, out_name, vocab_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    parser.add_argument("outputdir")
    args = parser.parse_args()

    make_file(Path(args.inputdir), Path(args.outputdir))
