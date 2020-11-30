#! /bin/bash

~/conda/envs/pl/bin/python	src/run_model.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	-p data/tomioka/transformer_config/vae_transformer_32.json \
	-o data/tomioka/output/transformer_vae/2020_11_30/2 \
	-l data/output/transformer_vae/2020_11_30/2
