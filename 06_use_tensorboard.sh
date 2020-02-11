DOCKER_IMAGE=cttsai1985/tensorflow-transformers

GPU_DEVICE='all'

SHM_SIZE=2G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

RootPort1=8888
DockerRootPort1=8888

RootPort2=6006
DockerRootPort2=6006

WORKDIR="/root/src/script"
LOG_DIR="../input/distilroberta-base_q384_a512"
docker rm $(docker ps -a -q)

CMD="tensorboard --logdir ${LOG_DIR} --bind_all"

echo run -i -t --gpus ${GPU_DEVICE} -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -p $RootPort2:$DockerRootPort2 -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR}$DOCKER_IMAGE $CMD

docker run -i -t --gpus ${GPU_DEVICE} -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -p $RootPort2:$DockerRootPort2 -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD
