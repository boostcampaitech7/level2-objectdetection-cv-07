# train.sh
source .env

# W&B 로그인
wandb login $WANDB_API_KEY

# WANDB_RUN_NAME
MODEL="retinanet"
RUN_SUFFIX="sj_1"
WANDB_RUN_NAME="mm_${MODEL}_${RUN_SUFFIX}"

# python train.py \
#     --model $MODEL \
#     --root '/data/ephemeral/home/data/dataset/' \
#     --classes "General trash" "Paper" "Paper pack" "Metal" "Glass" "Plastic" "Styrofoam" "Plastic bag" "Battery" "Clothing" \
#     --img_scale 512 512 \
#     --samples_per_gpu 4 \
#     --seed 2022 \
#     --gpu_ids 0 \
#     --num_classes 10 \
#     --grad_clip_max_norm 35 \
#     --grad_clip_norm_type 2 \
#     --max_keep_ckpts 3 \
#     --checkpoint_interval 1 \
#     --wandb_project $WANDB_PROJECT \
#     --wandb_run_name $WANDB_RUN_NAME \
#     --wandb_entity $WANDB_ENTITY



# chmod +x ./train.sh
# ./train.sh