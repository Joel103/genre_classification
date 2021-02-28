#!/bin/bash
# GPU environment
image="tensorflow/tensorflow:2.4.1-gpu-jupyter"
gpus="--gpus all"

# Laptop environment
#image=tensorflow/tensorflow:2.4.1-jupyter
#gpus=

container_name="dl4aed"
docker stop $container_name
docker rm $container_name

# --privileged=true : is needed for tensorboard profiling. Usually not recommended.
# -p 6006:6006 : is needed for tensorboard investigation
docker run -d --name $container_name -p 8888:8888 $gpus -v $(pwd)/:/tf/ $image

docker exec -it $container_name /bin/bash -c "apt update && apt install -y libsndfile1"

# printing the jupyter access token
sleep 5
docker logs $container_name
