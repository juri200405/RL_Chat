#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/tomioka/output/sentence_change/2021_01_15/ \
	--port 31096 --bind_all
