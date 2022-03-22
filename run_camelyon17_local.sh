python -m torch.distributed.launch --nproc_per_node=1 --master_port=50123 train.py \
        --batch 16 --iter 200000 --size 128  /home/jwu/Data/lmdb/camelyon17