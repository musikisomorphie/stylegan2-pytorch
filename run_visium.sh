CUDA_VISIBLE_DEVICES=2 taskset -c 64-95 python train.py \
   Data/Visium/GAN/crop \
    --data=Visium \
    --gene=61 \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --path_regularize=0.01 \
    --check_save=Data/Visium/GAN/dec0/