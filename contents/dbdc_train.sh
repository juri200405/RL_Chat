#! /bin/bash

~/conda/envs/pl/bin/python	src/dbdc_model.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/1/epoch009.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--data_file data/tomioka/dbdc_data.json \
	--output_dir data/tomioka/output/dbdc/2020_12_10/16/ \
	--gpu 2 \
	--lr 1e-4 \
	--num_epoch 150 \
	--gru_hidden 128 \
	--ff_hidden 64
