CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python train.py \
    Data/CosMx/GAN/crop \
    --data=CosMx \
    --gene=1000 \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --path_regularize=0.01 \
    --check_save=Data/CosMx/GAN/dec0/