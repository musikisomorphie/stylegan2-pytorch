python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 train.py \
    /root/Data/non_IID/decoder/rxrx19a_VERO_lmdb/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=1 \
    --check_save=/root/Data/non_IID/encoder/rxrx19a_VERO_chn1/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 train.py \
    /root/Data/non_IID/decoder/rxrx19a_HRCE_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=1 \
    --check_save=/root/Data/non_IID/encoder/rxrx19a_HRCE_chn1/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 train.py \
    /root/Data/non_IID/decoder/rxrx19a_VERO_lmdb/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=4 \
    --check_save=/root/Data/non_IID/encoder/rxrx19a_VERO_chn4/