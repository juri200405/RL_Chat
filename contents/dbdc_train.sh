#! /bin/bash

~/conda/envs/pl/bin/python	src/dbdc_model.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_11_27_3/0980k.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--data_file data/tomioka/dbdc_data.json \
	--output_dir data/tomioka/output/dbdc/2020_11_28/ \
	--gpu 2
