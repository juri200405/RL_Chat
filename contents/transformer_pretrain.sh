#! /bin/bash

python 	src/run_model.py \
	-m data/tomioka/without_kaomoji_model.model \
	-i data/tomioka/without_kaomoji.pkl \
	-p data/tomioka/transformer_config/vae_transformer.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/transformer_vae/2020_10_29_02/
