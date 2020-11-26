#! /bin/bash

python src/vae_check.py \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-i data/tomioka/no_kwdlc_126.pkl \
	-p data/tomioka/output/transformer_vae/2020_11_26/hyper_param.json \
	--pt_file data/tomioka/output/transformer_vae/2020_11_26/0030k.pt
