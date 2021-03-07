CUDA_VISIBLE_DEVICES=0 python train.py --name gta2cityscapes 
                                       --resize_or_crop='scale_width_and_crop' 
                                       --loadSize=1024 
                                       --fineSize=400 
                                       --model cycle_gan 
                                       --lambda_identity 1.0 
                                       --batchSize 1 
                                       --dataset_mode unaligned 
                                       --dataroot /mnt/data/cyclegan_data/ 
                                       --which_direction AtoB 
                                       --display_id=0 
                                       --checkpoints_dir='/ceph/cycada/cyclegan_checkpoints/' 
                                       --save_epoch_freq=1 
                                    #    --continue_train 
                                    #    --which_epoch=12 
                                    #    --epoch_count=12