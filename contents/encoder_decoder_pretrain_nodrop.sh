#! /bin/bash

python src/run_model.py \
	-m data/tomioka/laboroai_bert/large/webcorpus.model \
	-i data/tomioka/courpus.txt \
	-p data/tomioka/output/bert_vae/2020_08_24/hyper_param.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/bert_vae/2020_09_18/ \
	--pt_file data/tomioka/output/bert_vae/2020_08_24/cuda_epoch100.pt
