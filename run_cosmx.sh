python train.py \
    Data/CosMx/GAN/crop \
    --data=CosMx \
    --gene=1000 \
    --batch=16 \
    --iter=800000 \
    --size=128 \
    --channel=-1 \
    --kernel_size=5 \
    --gene_use \
    --split_label=slide_ID_numeric \
    --check_save=Data/CosMx/GAN/gik0/