#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/2/epoch009.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--grammar_data data/tomioka/grammar.json \
	--output_dir data/tomioka/output/sentence/2020_12_18/4/ \
	--num_epoch 1000 \
	--num_experiment 20 \
	--gpu 0
