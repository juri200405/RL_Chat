#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--output_dir data/output/sentence/2020_12_31/6/ \
	--mid_size 2048 \
	--num_epoch 10000 \
	--num_experiment 20 \
	--training_num 32 \
	--gpu 0 \
	--lr 1e-4 \
	--discount 0.0 \
	--initial_log_alpha -9.0 \
	--activation tanh \
	--additional_reward none \
	--no_gru \
	--random_state \
	--reward_checkpoint data/tomioka/output/is_in_corpus/2020_12_29/2/checkpoint.pt \
	# --grammar_data data/tomioka/grammar.json \
	# --manual_reward \
	# --use_history_hidden \
