#! /bin/bash

~/conda/envs/pl/bin/python	src/run_model.py \
	--spm_model data/tomioka/laboroai_bert/large/webcorpus.model \
	--bert_path data/tomioka/laboroai_bert/large/converted/ \
	--encoder_gpu 1 \
	--input_file data/tomioka/no_kwdlc_webcorpus_126.pkl \
	--hyper_param data/tomioka/transformer_config/bert_large.json \
	--output_dir data/tomioka/output/bert_vae/2020_12_31/1 \
	--log_dir data/output/bert_vae/2020_12_31/1
