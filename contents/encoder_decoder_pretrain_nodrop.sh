#! /bin/bash

python src/run_model.py \
	-m data/tomioka/laboroai_bert/large/webcorpus.model \
	-i data/tomioka/courpus.txt \
	-p data/tomioka/transformer_config/vae_2_2_Adam_nodrop.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/bert_vae/2020_09_02/
