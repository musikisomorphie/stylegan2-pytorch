CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/NSTG/GAN/ \
    --batch=16 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/NSTG/GAN/decoder0/

CUDA_VISIBLE_DEVICES=0 python train.py \
    /raid/jiqing/Data/NSTG/GAN/ \
    --batch=16 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/NSTG/GAN/decoder1/