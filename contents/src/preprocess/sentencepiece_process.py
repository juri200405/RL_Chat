from pathlib import Path
import argparse

import sentencepiece as spm

def make_spm_model(in_file, out_name, vocab_size):
    #spm.SentencePieceTrainer.train("--bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --input={} --model_prefix={} --vocab_size={} --add_dummy_prefix=False".format(in_file, out_name, vocab_size))
    spm.SentencePieceTrainer.train("--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --input={} --model_prefix={} --vocab_size={} --add_dummy_prefix=False".format(in_file, out_name, vocab_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    parser.add_argument("outputdir")
    parser.add_argument("output_name")
    args = parser.parse_args()

    make_spm_model(args.inputfile, str(Path(args.outputdir) / args.output_name), 8000)
