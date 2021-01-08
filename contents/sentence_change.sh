#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_change.py \
	--output_dir data/tomioka/output/sentence_change/2021_01_08/2/ \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--input_file data/tomioka/no_kwdlc_126.pkl \
	--ngram_file data/tomioka/no_kwdlc_3gram.pkl \
	--mid_size 2048 \
	--num_experiment 20 \
	--num_epoch 10000 \
	--training_num 32 \
	--gpu 1 \
	--lr 1e-4 \
	--discount 0.0 \
	--initial_log_alpha -9.0 \
	--activation sigmoid \
	--reward_type corpus_ngram \
	--additional_reward pos_state_action_cos \
