#! /bin/bash

~/conda/envs/pl/bin/python	src/is_in_corpus.py \
	--input_file data/tomioka/no_kwdlc_128.pkl
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--output_dir data/tomioka/output/is_in_corpus/2020_12_31/1/ \
	--gpu 0 \
	--num_data 1000000 \
	--num_epoch 10 \
	--score_noise
