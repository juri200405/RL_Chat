#! /bin/bash

python src/run_model.py \
	-m data/tomioka/laboroai_bert/large/webcorpus.model \
	-i data/tomioka/dbdc/all_utterances_noungram.txt \
	-p data/tomioka/transformer_config/vae_2_2_Adam_nodrop.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/output/bert_vae/2020_08_24/
