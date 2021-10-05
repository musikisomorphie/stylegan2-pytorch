# Setting up NVIDIA Container Toolkit
## Setup the stable repository and the GPG key:

>distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
Note

Install the nvidia-docker2 package (and dependencies) after updating the package listing:

```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

Restart the Docker daemon to complete the installation after setting the default runtime:
```
sudo systemctl restart docker
```
At this point, a working setup can be tested by running a base CUDA container:

```
sudo docker run --rm --gpus all nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
```

## Install Dockerfile
assume in the stylegan2 folder, run docker locally

```
docker build -t stylegan2 ./
```

run docker on dgx 
```
sudo docker build --build-arg http_proxy=http://proxy.usz.ch:8080 --build-arg https_proxy=http://proxy.usz.ch:8080 -t stylegan2 .
```

## Install ninja correctly (can be integrated in the dockerfile)
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
apt-get update
apt-get install unzip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

## run docker file 
```
docker run -it -v <path_to_code>:/root/code -v <path_to_data>:/root/data --gpus all stylegan2
```

## clean docker files
```
docker system prune
```

## run stylegan2

