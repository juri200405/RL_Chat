#! /bin/bash

python 	src/run_model.py \
	-m data/tomioka/courpus_model.model \
	-i data/tomioka/courpus_uniq.txt \
	-p data/tomioka/transformer_config/vae_transformer.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/transformer_vae/2020_10_17_02/
