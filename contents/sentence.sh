#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/2/epoch009.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--grammar_data data/tomioka/grammar.json \
	--output_dir data/tomioka/output/sentence/2020_12_24_2/3/ \
	--num_epoch 1000 \
	--num_experiment 20 \
	--gpu 2 \
	--lr 2e-5 \
	--discount 0.0 \
	--initial_log_alpha 1e-4 \
	--training_num 16 \
	--activation tanh \
	--additional_reward none \
	--use_history_hidden \
	--no_gru \
