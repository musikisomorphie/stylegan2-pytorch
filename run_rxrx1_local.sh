python -m torch.distributed.launch --nproc_per_node=1 --master_port=50123 train.py \
        --batch 8 /root/rxrx1/