# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org"
ARG GIT_CLONE="git clone --depth 10"

ENV HOME /root

WORKDIR $HOME

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive

RUN $APT_INSTALL build-essential software-properties-common ca-certificates \
                 wget git zlib1g-dev nasm cmake 

RUN $GIT_CLONE https://github.com/libjpeg-turbo/libjpeg-turbo.git
WORKDIR libjpeg-turbo
RUN mkdir build
WORKDIR build
RUN cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=libjpeg-turbo -DWITH_JPEG8=1 ..
RUN make
RUN make install
WORKDIR libjpeg-turbo
RUN mv include/jerror.h include/jmorecfg.h include/jpeglib.h include/turbojpeg.h /usr/include
RUN mv include/jconfig.h /usr/include/x86_64-linux-gnu
RUN mv lib/*.* /usr/lib/x86_64-linux-gnu
RUN mv lib/pkgconfig/* /usr/lib/x86_64-linux-gnu/pkgconfig
RUN ldconfig
WORKDIR HOME

RUN apt-key adv --keyserver keyserver.ubuntu.com/ --recv-keys BA6932366A755776 
# --keyserver-options http-proxy=http://proxy.usz.ch:8080
RUN add-apt-repository 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main'
# # RUN deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main
# RUN wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tar.xz

RUN apt-get update

RUN $APT_INSTALL python3.7 python3.7-dev

RUN wget -O $HOME/get-pip.py https://bootstrap.pypa.io/get-pip.py --no-check-certificate

RUN python3.7 $HOME/get-pip.py 

RUN ln -s /usr/bin/python3.7 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN $PIP_INSTALL setuptools 
RUN $PIP_INSTALL numpy scipy nltk lmdb cython pydantic pyhocon 

RUN $PIP_INSTALL torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing"

RUN python -m pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo 
RUN CFLAGS="${CFLAGS} -mavx2" $PIP_INSTALL --force-reinstall --no-binary :all: --compile pillow-simd 

RUN $APT_INSTALL libsm6 libxext6 libxrender1
RUN $PIP_INSTALL opencv-python-headless 

WORKDIR $HOME

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
RUN apt-get update
RUN $APT_INSTALL unzip
RUN unzip ninja-linux.zip -d /usr/local/bin/
RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

RUN ldconfig
RUN apt-get clean
RUN apt-get autoremove
RUN rm -rf /var/lib/apt/lists/* /tmp/* ~/*