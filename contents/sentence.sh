#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_rl.py \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--output_dir data/output/sentence/2020_12_27/5/ \
	--mid_size 2048 \
	--num_epoch 5000 \
	--num_experiment 20 \
	--training_num 32 \
	--gpu 0 \
	--lr 1e-3 \
	--discount 0.0 \
	--initial_log_alpha 0.0 \
	--activation tanh \
	--additional_reward cos \
	# --use_history_hidden \
	# --no_gru \
	# --random_state \
	# --manual_reward \
	# --grammar_data data/tomioka/grammar.json \
