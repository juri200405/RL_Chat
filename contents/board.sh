#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/output/transformer_vae/2021_01_17/ \
	--port 31096 --bind_all
