from mmcv import Config
from data_utils import load_annotations, split_data, prepare_datasets
from config_utils import load_config, update_config
from train_utils import build_and_train_model

# Configuration
root='/data/ephemeral/home/dataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# Load annotations
data = load_annotations(f'{root}/train.json')

# Train/validation split
train_data, val_data = split_data(data)

# Prepare datasets
prepare_datasets(root, data, train_data, val_data)

# Load and update config
cfg = load_config('mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py')
cfg = update_config(cfg, root, classes)

# Build and train model
build_and_train_model(cfg)