FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip==9.0.3 \
    && pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl torchvision \
    && rm -rf ~/.cache/pip

WORKDIR /workspace