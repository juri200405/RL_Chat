#! /bin/bash

python 	src/bert_data.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc.txt \
	-o data/tomioka/no_kwdlc_126.pkl \
	-n 1 \
	-l 126
