CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=0 \
    --check_save=/raid/jiqing/Data/non_IID/encoder/rxrx19b_HUVEC_chn0/

CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=1 \
    --check_save=/raid/jiqing/Data/non_IID/encoder/rxrx19b_HUVEC_chn1/
