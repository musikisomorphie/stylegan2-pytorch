CUDA_VISIBLE_DEVICES=5 taskset -c 160-191 python train.py \
   Data/Visium/GAN/crop \
    --data=Visium \
    --gene=61 \
    --batch=8 \
    --iter=200000 \
    --size=128 \
    --channel=-1 \
    --check_save=Data/Visium/GAN/decoder0/