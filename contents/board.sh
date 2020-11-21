#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/transformer_vae/2020_11_21/ \
	--port 31096 --bind_all
