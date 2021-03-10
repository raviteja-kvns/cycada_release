
gpu=1

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=100000
crop=768
snapshot=5000
batch=1

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='cyclegta5'
tgt='cityscapes'
datadir='./../data/'

resdir="results/${src}_to_${tgt}/adda_ups_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
model='upsnet'
baseiter=115000
#model='fcn8s'
#baseiter=100000


# base_model="base_models/${model}-${src}-iter${baseiter}.pth"
base_model="base_models/upsnet_resnet_50_gta5_121000.pth"

outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

# UPSNet config
ups_net_config='../UPSNet/upsnet/experiments/upsnet_resnet50_gta5.yaml'

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/train_ups_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --ups_net_config ${ups_net_config} \
    --lr ${lr} --momentum ${momentum} --gpu 0 \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot
