#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/sentence/2020_12_31/ \
	--port 31096 --bind_all
