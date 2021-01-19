#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_simple.py \
	--output_dir data/output/sentence_change/2021_01_19/1/ \
	--vae_checkpoint data/tomioka/output/transformer_vae/2021_01_17/1/epoch002.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--input_file data/tomioka/no_kwdlc_126.pkl \
	--mid_size 2048 \
	--num_experiment 20 \
	--num_epoch 20000 \
	--training_num 64 \
	--gpu 0 \
	--lr 1e-4 \
	--discount 0.0 \
	--initial_log_alpha 0.0 \
	--reward_type token_len_reward \
	--target_len 10 \
	--target_range 1 \
	--repeat_num 3
