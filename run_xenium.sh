CUDA_VISIBLE_DEVICES=1 taskset -c 32-63 python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --check_save=Data/Xenium/GAN/g0/

CUDA_VISIBLE_DEVICES=1 taskset -c 32-63 python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --split_scheme=rep \
    --gene_use --meta_use \
    --check_save=Data/Xenium/GAN/g0/