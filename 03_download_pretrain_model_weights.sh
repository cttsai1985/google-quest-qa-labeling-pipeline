DOCKER_IMAGE=cttsai1985/tensorflow-transformers

GPU_DEVICE='"device=0"'

SHM_SIZE=2G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

RootPort1=8888
DockerRootPort1=8888

RootPort2=6666
DockerRootPort2=6666

WORKDIR="/root/src/script"

docker rm $(docker ps -a -q)

CMD="python download_pretrained.py"

echo run -i -t --gpus ${GPU_DEVICE} -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR}$DOCKER_IMAGE $CMD

docker run -i -t --gpus ${GPU_DEVICE} -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD
