#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/tomioka/output/sentence/2020_12_25/ \
	--port 31096 --bind_all
