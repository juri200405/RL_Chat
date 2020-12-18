#! /bin/bash

~/conda/envs/pl/bin/python	src/dbdc_model.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/2/epoch009.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--data_file data/tomioka/dbdc_data.json \
	--output_dir data/tomioka/output/dbdc/2020_12_17/6/ \
	--gpu 1 \
	--lr 5e-5 \
	--num_epoch 150 \
	--gru_hidden 128 \
	--ff_hidden 128 \
	--transformer True
