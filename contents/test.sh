#! /bin/bash

~/conda/envs/pl/bin/python src/chat_main.py \
	-s data/tomioka/telegram_setting.json \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-o data/tomioka/output/rl_chat/2020_12_08 \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/2/5160k.pt \
	--database data/tomioka/database2.json \
	--batch_size 64 \
	--sample_size 512 \
	--telegram
