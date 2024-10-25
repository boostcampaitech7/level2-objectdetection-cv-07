#!/bin/bash

# 환경 변수 설정 (필요시 추가)
export PYTHONPATH=/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection:$PYTHONPATH

# 기본 변수 설정
CONFIG_PATH="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py"  # config 파일 경로
CHECKPOINT_FILE="epoch_12.pth"           # checkpoint 파일 이름
WORK_DIR="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/work_dirs/co_dino"      # 작업 디렉터리 (모델 저장, 평가 파일 저장)
DATA_ROOT="/data/ephemeral/home/data/dataset"  # 데이터셋 루트 경로
TEST_ANN_FILE="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/valid_0_5.json"  # 테스트 어노테이션 파일

# Python 스크립트를 실행하여 모델을 테스트하고 결과를 submission.csv로 저장
python ./tools/inference.py \
    $CONFIG_PATH \
    $CHECKPOINT_FILE \
    --work-dir $WORK_DIR \
    --data_root $DATA_ROOT \
    --test_ann_file $TEST_ANN_FILE \
    --tta \
    --eval bbox

# 옵션 설명:d
# --show-dir: 예측 이미지를 저장할 디렉토리 (선택사항)
# --wait-time: 이미지 표시 간격
# --skip: 평가 생략
# --launcher: 분산 런처 옵션 (필요시 슬럼이나 MPI 선택 가능)

# chmod +x ./tools/inference_valid.sh
# ./tools/inference_valid.sh
