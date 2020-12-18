#! /bin/bash

~/conda/envs/pl/bin/tensorboard \
	--logdir data/tomioka/output/dbdc/2020_12_18/ \
	--port 31096 --bind_all
