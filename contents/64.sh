#! /bin/bash

~/conda/envs/pl/bin/python	src/run_model.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	-p data/tomioka/output/transformer_vae/2020_12_26/1/hyper_param.json \
	-o data/tomioka/output/transformer_vae/2021_01_17/1 \
	-l data/output/transformer_vae/2021_01_17/1
