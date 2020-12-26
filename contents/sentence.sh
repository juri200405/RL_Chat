#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/2/epoch009.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--output_dir data/output/sentence/2020_12_26/5/ \
	--mid_size 2048 \
	--num_epoch 5000 \
	--num_experiment 20 \
	--gpu 1 \
	--lr 1e-3 \
	--discount 0.0 \
	--initial_log_alpha 0.0 \
	--training_num 32 \
	--activation tanh \
	--additional_reward state_action_cos \
	--no_gru \
	--random_state \
	# --grammar_data data/tomioka/grammar.json \
	# --use_history_hidden \
