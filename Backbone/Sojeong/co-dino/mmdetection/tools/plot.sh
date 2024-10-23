#!/bin/bash

export PYTHONPATH=/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection:$PYTHONPATH

# 실행할 Python 스크립트 경로
PYTHON_SCRIPT="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/tools/visualization.py"  # COCO 평가 및 PR Curve 그리는 스크립트
# 저장된 pkl 파일 경로
PKL_FILE="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/results_2.pkl"
# 테스트할 데이터셋의 annotation 파일 경로
ANN_FILE="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/valid_0_5.json"
# 결과를 저장할 경로
WORK_DIR="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/work_dirs/co_dino/plot"

# Python 스크립트 실행
python $PYTHON_SCRIPT $PKL_FILE $ANN_FILE --work-dir $WORK_DIR

# chmod +x tools/plot.sh
# ./tools/plot.sh
