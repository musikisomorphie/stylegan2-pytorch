CUDA_VISIBLE_DEVICES=3 taskset -c 96-127 python train.py \
   Data/Visium/GAN/crop \
    --data=Visium \
    --gene=61 \
    --batch=8 \
    --iter=400000 \
    --size=128 \
    --channel=-1 \
    --kernel_size=5 \
    --gene_use --meta_use \
    --check_save=Data/Visium/GAN/g0/