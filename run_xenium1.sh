python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --check_save=Data/Xenium/GAN/g0/

python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --check_save=Data/Xenium/GAN/g1/