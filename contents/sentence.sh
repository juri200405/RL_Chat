#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--reward_checkpoint data/tomioka/output/is_in_corpus/2020_12_29/2/checkpoint.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--output_dir data/tomioka/output/sentence/2020_12_30/3/ \
	--mid_size 2048 \
	--num_epoch 5000 \
	--num_experiment 20 \
	--training_num 32 \
	--gpu 2 \
	--lr 1e-3 \
	--discount 0.0 \
	--initial_log_alpha 0.0 \
	--activation tanh \
	--additional_reward none \
	--no_gru \
	--random_state \
	--grammar_data data/tomioka/grammar.json \
	# --manual_reward \
	# --use_history_hidden \
