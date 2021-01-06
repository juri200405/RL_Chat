#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/bert_vae/2020_12_31/ \
	--port 31096 --bind_all
