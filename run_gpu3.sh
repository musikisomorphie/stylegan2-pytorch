CUDA_VISIBLE_DEVICES=3 python -m train.py \
    Data/non_IID/decoder/rxrx19b_HUVEC_lmdb/ \
    --batch=8 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --check_save=/root/Data/non_IID/encoder/rxrx19b_HUVEC/
