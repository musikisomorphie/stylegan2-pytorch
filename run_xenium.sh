python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --split_label=kmeans_2_clusters \
    --check_save=Data/Xenium/GAN/gl0/

python train.py \
   Data/Xenium/GAN/crop \
    --data=Xenium \
    --gene=280 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --gene_use \
    --split_label=kmeans_2_clusters \
    --check_save=Data/Xenium/GAN/gl1/