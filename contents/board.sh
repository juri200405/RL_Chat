#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/tomioka/output/rl_chat/2020_12_08/ \
	--port 31096 --bind_all
