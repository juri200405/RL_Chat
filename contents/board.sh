#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/sentence_change/2021_01_16/ \
	--port 31096 --bind_all
