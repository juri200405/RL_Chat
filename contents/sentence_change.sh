#! /bin/bash

~/conda/envs/pl/bin/python	src/sentence_change.py \
	--output_dir data/output/sentence_change/2021_01_16/1/ \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_26/1/epoch000.pt \
	--input_file data/tomioka/no_kwdlc_126.pkl \
	--spm_model data/tomioka/spm_model/no_kwdlc.model \
	--datas_file data/tomioka/no_kwdlc.txt \
	--mid_size 2048 \
	--num_experiment 20 \
	--num_epoch 10000 \
	--training_num 32 \
	--gpu 0 \
	--lr 1e-4 \
	--discount 0.0 \
	--initial_log_alpha 0.0 \
	--activation sigmoid \
	--reward_type sentence_head_reward \
	--additional_reward none \
	# --additional_reward_rate 0.7 \
	# --target_len 30 \
	# --ngram_file data/tomioka/no_kwdlc_3gram.pkl \
