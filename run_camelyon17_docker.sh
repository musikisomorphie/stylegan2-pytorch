sudo docker run -it -v /raid/jiqing/Github/stylegan2-pytorch/:/root/stylegan2 -v /raid/jiqing/Data/non_IID/decoder/camelyon17_lmdb/:/root/camelyon17 --gpus '"device=0,1"'  stylegan2_11.1