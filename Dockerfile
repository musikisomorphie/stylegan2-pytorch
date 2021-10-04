# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV https_proxy=http://proxy.usz.ch:8080/
ENV http_proxy=http://proxy.usz.ch:8080/
ENV HTTP_PROXY=http://proxy.usz.ch:8080/
ENV HTTPS_PROXY=http://proxy.usz.ch:8080/
ENV no_proxy="localhost,localdomain,127.0.0.1"
ENV NO_PROXY="localhost,localdomain,127.0.0.1"

ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
ARG GIT_CLONE="git clone --depth 10"

ENV HOME /root

WORKDIR $HOME

RUN apt-get update \
 && apt-get install -y sudo

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

# this is where I was RUN sudoning into problems with the other approaches
RUN sudo apt-get update 

RUN sudo rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list



ARG DEBIAN_FRONTEND=noninteractive

RUN sudo $APT_INSTALL build-essential software-properties-common ca-certificates \
                 wget git zlib1g-dev nasm cmake unzip

RUN sudo $GIT_CLONE https://github.com/libjpeg-turbo/libjpeg-turbo.git
WORKDIR libjpeg-turbo
RUN sudo mkdir build
WORKDIR build
RUN sudo cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=libjpeg-turbo -DWITH_JPEG8=1 ..
RUN sudo make
RUN sudo make install
WORKDIR libjpeg-turbo
RUN sudo mv include/jerror.h include/jmorecfg.h include/jpeglib.h include/turbojpeg.h /usr/include
RUN sudo mv include/jconfig.h /usr/include/x86_64-linux-gnu
RUN sudo mv lib/*.* /usr/lib/x86_64-linux-gnu
RUN sudo mv lib/pkgconfig/* /usr/lib/x86_64-linux-gnu/pkgconfig
RUN sudo ldconfig
WORKDIR HOME

RUN sudo -E add-apt-repository ppa:deadsnakes/ppa

RUN sudo apt-get update

RUN sudo $APT_INSTALL python3.7 python3.7-dev

RUN sudo wget -O $HOME/get-pip.py https://bootstrap.pypa.io/get-pip.py

RUN sudo python3.7 $HOME/get-pip.py

RUN sudo ln -s /usr/bin/python3.7 /usr/local/bin/python3
RUN sudo ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN sudo $PIP_INSTALL setuptools
RUN sudo $PIP_INSTALL numpy scipy nltk lmdb cython pydantic pyhocon

RUN sudo $PIP_INSTALL torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing"

# RUN sudo $PIP_INSTALL 'git+https://github.com/facebookresearch/detectron2.git'

RUN sudo python -m pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo
RUN sudo CFLAGS="${CFLAGS} -mavx2" $PIP_INSTALL --force-reinstall --no-binary :all: --compile pillow-simd

RUN sudo $APT_INSTALL libsm6 libxext6 libxrender1
RUN sudo $PIP_INSTALL opencv-python-headless

# WORKDIR $HOME
# RUN sudo $GIT_CLONE https://github.com/NVIDIA/apex.git
# WORKDIR apex
# RUN sudo $PIP_INSTALL -v --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# WORKDIR $HOME
# RUN sudo $GIT_CLONE https://github.com/cocodataset/cocoapi.git
# WORKDIR cocoapi/PythonAPI
# RUN sudo make
# RUN sudo python setup.py build_ext install

WORKDIR $HOME

RUN sudo wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
RUN sudo unzip ninja-linux.zip -d /usr/local/bin/
RUN sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
RUN sudo ldconfig
RUN sudo apt-get clean
RUN sudo apt-get autoremove
RUN sudo rm -rf /var/lib/apt/lists/* /tmp/* ~/*