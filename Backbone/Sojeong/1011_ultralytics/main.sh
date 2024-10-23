#!/bin/bash
# Shell script to run YOLOv8 train or inference

MODE=$1

if [ "$MODE" == "train" ]; then
    python3 main.py --mode train
elif [ "$MODE" == "inference" ]; then
    python3 main.py --mode inference
else
    echo "Invalid mode. Please use 'train' or 'inference'."
fi

# train mode : bash main.sh train
# inference mode : bash main.sh inference