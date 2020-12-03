#! /bin/bash

python src/chat_main.py \
	-s data/tomioka/telegram_setting.json \
	-m data/tomioka/spm_model/no_kwdlc.model \
	-o data/tomioka/output/rl_chat/2020_12_04 \
	--vae_checkpoint data/tomioka/output/transformer_vae/2020_12_02/3/0400k.pt \
	--bert_path data/tomioka/laboroai_bert/large/converted/ \
	--database data/tomioka/database.json \
	--batch_size 64 \
	--sample_size 128
