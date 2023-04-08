CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python train.py \
    Data/CosMx/GAN/crop \
    --data=CosMx \
    --gene=1000 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --check_save=Data/CosMx/GAN/g0/

CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 python train.py \
    Data/CosMx/GAN/crop \
    --data=CosMx \
    --gene=1000 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use --meta_use \
    --check_save=Data/CosMx/GAN/g0/