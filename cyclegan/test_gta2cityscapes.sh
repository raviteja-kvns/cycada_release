CUDA_VISIBLE_DEVICES=0 python test.py --name gta2cityscapes \
    --resize_or_crop="crop" \
    --loadSize=1920 --fineSize=1920 \
    --how_many 50 \
    --which_epoch 20 \
    --model cycle_gan \
    --batchSize 1 \
    --dataset_mode unaligned --dataroot /mnt/data/cyclegan_data/ \
    --which_direction AtoB
