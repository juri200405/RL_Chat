#! /bin/bash

~/conda/envs/pl/bin/python src/chat_main.py \
	-s data/tomioka/telegram_setting.json \
	-o data/tomioka/output/rl_chat/2020_12_11 \
	--dbdc_checkpoint data/tomioka/output/dbdc/2020_12_10/13/epoch063.pt \
	--batch_size 64 \
	--sample_size 512 \
	# --telegram
	# --database data/tomioka/database2.json \
