#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/sentence/2020_12_27/ \
	--port 31096 --bind_all
