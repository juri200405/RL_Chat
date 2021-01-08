#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/tomioka/output/sentence_change/2021_01_08/ \
	--port 31096 --bind_all
