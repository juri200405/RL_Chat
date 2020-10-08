#! /bin/bash

python src/vae_check.py \
	-m data/tomioka/laboroai_bert/large/webcorpus.model \
	-i data/tomioka/dbdc/all_utterances_noungram.txt \
	-p data/tomioka/output/bert_vae/2020_08_24/hyper_param.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/bert_vae/2020_09_18/ \
	--pt_file data/tomioka/output/bert_vae/2020_08_24/cuda_epoch010.pt
