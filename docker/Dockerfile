ARG CONTAINER_IMAGE
FROM $CONTAINER_IMAGE

LABEL maintainer "Chia-Ta Tsai <cttsai.aws@outllok.com>"

########### LOCAL DOCKER INFO
ARG CUDA_VERSION_CURRENT=10.1
ARG PYTHON_VERSION_SHORT=3.6

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN ln -sf /usr/share/zoneinfo/US/Eastern /etc/localtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# use for CUDA
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib

# system packages
RUN apt-get update && apt-get install -y --no-install-recommends htop tmux && rm -rf /var/lib/apt/lists/*


########### pip
COPY requirements_pip.txt /tmp/requirements_pip.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /tmp/requirements_pip.txt

# clean up
COPY ./clean-layer.sh /tmp/clean-layer.sh
RUN bash /tmp/clean-layer.sh


########### General
EXPOSE 6006 8888

VOLUME ["/log"]
VOLUME ["/root/src"]
VOLUME ["/root/src/input"]
VOLUME ["/root/src/script"]

# COPY jupyter_notebook_config.py /root/.jupyter/

WORKDIR /root/src
# CMD ["sh", "-c", "jupyter notebook --port=8888 --no-browser --ip=* --allow-root"]
