#! /bin/bash

~/conda/envs/pl/bin/python	src/run_model.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	-p data/tomioka/transformer_config/128.json \
	-o data/tomioka/output/transformer_vae/2020_12_02/1 \
	-l data/output/transformer_vae/2020_12_02/1
