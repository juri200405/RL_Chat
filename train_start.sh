#!/bin/bash

is_docker=false
for item in $(id -Gn)
do
	if [ ${item} = "docker" ]; then
		is_docker=true
	fi
done


if ${is_docker}; then
	docker-compose down
else
	sudo docker-compose down
fi

if [ -f ./.env ]; then
	rm ./.env
fi

echo http_proxy >> ./.env
echo https_proxy >> ./.env
echo HTTP_PROXY >> ./.env
echo HTTPS_PROXY >> ./.env
echo "U_ID=$(id -u)" >> ./.env
echo "G_ID=$(id -g)" >> ./.env
echo "HOME_DIR="$HOME >> ./.env
echo "USER_NAME=$(id -un)" >> ./.env

if ${is_docker}; then
	docker-compose up -d
	docker-compose exec -d tomioka_chat $@
	docker-compose exec -d tomioka_chat ./board.sh
else
	sudo -E docker-compose up -d
	sudo -E docker-compose exec -d tomioka_chat $@
	sudo -E docker-compose exec -d tomioka_chat ./board.sh
fi

exit 0
