CUDA_VISIBLE_DEVICES=3 python train.py \
    /raid/jiqing/Data/NSTG/GAN/ \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=/raid/jiqing/Data/NSTG/GAN/decoder3/