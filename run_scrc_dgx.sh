python -m torch.distributed.launch --nproc_per_node=2 --master_port=54567 train.py \
        --batch 16 /root/scrc_012/