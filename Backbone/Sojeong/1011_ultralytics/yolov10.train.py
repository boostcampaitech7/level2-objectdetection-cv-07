import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from dotenv import load_dotenv
import wandb

import os
from ultralytics import YOLO
from dotenv import load_dotenv
import wandb

# Load .env file
load_dotenv()

# W&B login
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    raise ValueError("WANDB_API_KEY not found in .env file")

# Initialize WandB
wandb.init(project=wandb_project, entity=wandb_entity, name="yolov10")

# Load a model
model = YOLO("/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/yolov10x.pt")

# Add W&B callback for Ultralytics
step = 0
def wandb_callback(batch, logs=None, **kwargs):
    global step
    if logs is None:
        logs = {}  # logs가 전달되지 않으면 빈 딕셔너리로 설정
    wandb.log(logs, step=step)  # Log metrics with manual step control
    step += 1  # Increment step after each batch

model.add_callback("on_train_batch_end", wandb_callback)

# Train the model
train_results = model.train(
    data="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/coco_dataset.yaml",  # path to COCO dataset YAML
    epochs=100,  # number of training epochs
    patience=30, # early stopping patience
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    amp=False,
    batch=8,
    name="yolov10_iou0.7",  # 새로운 실행에 대한 이름 지정
    exist_ok=False,  # 동일한 이름의 디렉토리가 있을 때 덮어쓰지 않도록 설정
    save_dir="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/runs/detect"
)

# Finish the W&B run
wandb.finish()

# python /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/yolov10.train.py