#!/bin/bash

CURRENT_USER=$(whoami)
CONTAINER_NAME=$1
ID=`id -u`

if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME=$CURRENT_USER"_roll_dev"
    echo "No container name specified; using the default name: $CONTAINER_NAME"
fi

echo "current user: $CURRENT_USER"

# W&B credentials live in the host environment, never in the image or the repo.
# Without this, exporting WANDB_API_KEY before running this script has no effect
# inside the container, and any config with track_with=wandb dies at wandb.init
# with "No API key configured" (the repo's own configs pass no api_key and rely
# on the ambient variable).  Only variables the host actually sets are
# forwarded, so an unset one still falls back to wandb's own defaults.
WANDB_ENV=()
for v in WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE; do
    if [ -n "${!v}" ]; then
        WANDB_ENV+=(-e "$v=${!v}")
    fi
done
if [ -z "$WANDB_API_KEY" ]; then
    echo "warning: WANDB_API_KEY is not set on the host; training with track_with=wandb will fail inside the container" >&2
fi

sudo docker run -dit \
     "${WANDB_ENV[@]}" \
     --cap-add SYS_ADMIN \
     --gpus all \
     --name=$CONTAINER_NAME \
     --ipc=host \
     --net=host \
     -v /dev/fuse:/dev/fuse \
     -v /dev/shm:/dev/shm \
     -v /home/$CURRENT_USER/:/home/$CURRENT_USER/ \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v /mnt/ram:/mnt/ram \
     roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084 \
     /bin/bash

sudo docker exec -i $CONTAINER_NAME groupadd sdev
sudo docker exec -i $CONTAINER_NAME /usr/sbin/useradd -MU -G sdev -u $ID $CURRENT_USER
sudo docker exec -it $CONTAINER_NAME /usr/bin/su $CURRENT_USER
