#! /bin/bash

python 	src/bert_data.py \
	-m data/tomioka/normalize_model.model \
	-i data/tomioka/normalize.txt \
	-o data/tomioka/normalize.pkl
