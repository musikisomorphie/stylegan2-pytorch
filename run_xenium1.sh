CUDA_VISIBLE_DEVICES=1 taskset -c 32-63 python train.py \
    /raid/jiqing/Data/Xenium/GAN/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/Data/Xenium/GAN/decoder2/

CUDA_VISIBLE_DEVICES=1 taskset -c 32-63 python train.py \
    /raid/jiqing/Data/Xenium/GAN/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/Data/Xenium/GAN/decoder3/