#! /bin/bash

~/conda/envs/pl/bin/python	src/pl_run.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	-p data/tomioka/transformer_config/vae_transformer.json \
	-b data/tomioka/laboroai_bert/large/converted/ \
	-o data/tomioka/output/transformer_vae/2020_11_25/ \
	-l data/output/transformer_vae/2020_11_25/ \
	--eos_weight 1.0 \
	--gpus 2 \
	--val_check_interval 0.1
