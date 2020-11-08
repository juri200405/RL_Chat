#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/tomioka/output/transformer_vae/2020_11_08/ \
	--port 31096 --bind_all
