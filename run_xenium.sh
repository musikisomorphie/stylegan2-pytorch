CUDA_VISIBLE_DEVICES=1 taskset -c 32-63 python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --check_save=Data/Xenium/GAN/decoder0/