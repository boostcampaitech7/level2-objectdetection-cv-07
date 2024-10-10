#!/bin/bash

CONFIG_PATH='mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py'
ROOT='../../../dataset/'
EPOCH='latest'

python mmdetection/inference.py --config_path $CONFIG_PATH --root $ROOT --epoch $EPOCH

# chmod +x mmdetection/inference.sh