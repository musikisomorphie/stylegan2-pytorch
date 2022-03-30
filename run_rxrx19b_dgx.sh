python -m torch.distributed.launch --nproc_per_node=2 --master_port=50123 train.py \
        --batch 32 --iter 200000 --size 128  /root/rxrx19b