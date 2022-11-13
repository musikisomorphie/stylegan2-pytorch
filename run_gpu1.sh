CUDA_VISIBLE_DEVICES=1 python train.py \
    /raid/jiqing/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=2 \
    --check_save=/raid/jiqing/Data/non_IID/encoder/rxrx19b_HUVEC_chn2/

CUDA_VISIBLE_DEVICES=1 python train.py \
    /raid/jiqing/Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=3 \
    --check_save=/raid/jiqing/Data/non_IID/encoder/rxrx19b_HUVEC_chn3/
