python -m torch.distributed.launch --nproc_per_node=1 --master_port=12341 train.py \
    /root/Data/non_IID/decoder/rxrx19b_cell_lmdb/
    --batch 8 
    --iter 800000 
    --size 128    
    --channel 1
    --check_save /root/Data/non_IID/encoder/rxrx19b_cell/chn1/  