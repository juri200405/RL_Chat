#! /bin/bash

~/conda/envs/pl/bin/python src/vae_check.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	--pt_file data/tomioka/output/transformer_vae/2020_12_02/2/epoch009.pt
