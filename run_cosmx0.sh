CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/CosMx/GAN/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/Data/CosMx/GAN/decoder0/

CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/CosMx/GAN/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/Data/CosMx/GAN/decoder1/