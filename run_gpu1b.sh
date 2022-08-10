python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 train.py \
    /root/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=2 \
    --check_save=/root/Data/non_IID/encoder/rxrx19b_HUVEC_chn2/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 train.py \
    /root/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=3 \
    --check_save=/root/Data/non_IID/encoder/rxrx19b_HUVEC_chn3/