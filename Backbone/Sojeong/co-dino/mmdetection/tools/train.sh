# 환경 변수 설정
export PYTHONPATH=/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection:$PYTHONPATH

# 훈련 스크립트 실행
python ./tools/train.py /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py \
    --classes "General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing" \
    --data_root /data/ephemeral/home/data/dataset \
    --ann_file /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/train_1_5.json \
    --batch_size 2 \
    --num_classes 10 \
    --work_dir /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/work_dirs/co_dino/valid1 \
    --eval_ann_file /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/valid_1_5.json \
    --image_size 512 512 \
    --num_workers 2 \

    # epoch 등 다시 설정

# chmod +x tools/train.sh
# ./tools/train.sh