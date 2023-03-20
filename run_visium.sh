CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python train.py \
   Data/Visium/GAN/crop \
    --data=Visium \
    --gene=61 \
    --batch=8 \
    --iter=200000 \
    --size=128 \
    --channel=-1 \
    --check_save=Data/Visium/GAN/decoder0/