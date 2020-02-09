#!/bin/bash

IMG_REPO=cttsai1985/tensorflow-transformers:latest
BASE_IMAGE=tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

echo docker build --rm -t ${IMG_REPO} --build-arg CONTAINER_IMAGE=${BASE_IMAGE} -f Dockerfile .
cd docker; docker build --rm -t ${IMG_REPO} --build-arg CONTAINER_IMAGE=${BASE_IMAGE} -f Dockerfile .

