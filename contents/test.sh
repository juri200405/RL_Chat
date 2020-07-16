#! /bin/bash

python src/chat_main.py \
	-s data/tomioka/telegram_setting.json \
	-m data/tomioka/laboroai_bert/large/webcorpus.model \
	-p data/tomioka/chat_hp.json \
	-o data/tomioka/output/rl_chat/2020_07_14 \
	--pt_file data/tomioka/output/bert_gru_transformer/2020_06_05/cuda_epoch199.pt \
	--bert_path data/tomioka/laboroai_bert/large/converted/ \
	--database data/tomioka/database.json \
	$@
