#!/bin/bash
container_name="dl4aed"
docker stop $container_name
docker rm $container_name

# --privileged=true : is needed for tensorboard profiling. Usually not recommended.
# -p 6006:6006 : is needed for tensorboard investigation

# GPU environment
#docker run -d --name $container_name -p 8888:8888 --gpus all -v $(pwd)/:/tf/ tensorflow/tensorflow:2.4.1-gpu-jupyter

# Laptop environment
docker run -d --name $container_name -p 8888:8888 -v $(pwd)/:/tf/ tensorflow/tensorflow:2.4.1-jupyter

sleep 3
docker logs $container_name
