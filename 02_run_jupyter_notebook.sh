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

WORKDIR="/root/src/notebook"

docker rm $(docker ps -a -q)

CMD="jupyter notebook --port ${DockerRootPort1} --ip=0.0.0.0 --allow-root --no-browser"

echo docker run -i -t --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD

docker run -i -t --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v $RootSrcPath:$DockerRootSrcPath -v $(readlink -f $DataPath):$DockerDataPath --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD
