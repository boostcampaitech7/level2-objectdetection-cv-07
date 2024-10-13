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
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    raise ValueError("WANDB_API_KEY not found in .env file")

# Define the epoch as the x-axis (step) for W&B metrics
wandb.define_metric("epoch", step_metric="epoch")

# Load a model
model = YOLO("yolov8x.pt")

# Initialize WandB
wandb.init(project="object-detection", entity="luckyvicky", name="yolov8")

# Add W&B callback for logging at the end of each batch
model.add_callback("on_train_batch_end", lambda trainer: wandb.log(
    {"batch_loss": trainer.loss, "epoch": trainer.epoch}, step=trainer.epoch * len(trainer.dataloader) + trainer.batch
))

# Train the model
train_results = model.train(
    data="/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/coco_dataset.yaml",  # path to COCO dataset YAML
    epochs=100,  # number of training epochs
    patience=30, # early stopping patience
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    amp=False,
    name="exp1",  # 새로운 실행에 대한 이름 지정
    exist_ok=False           # 동일한 이름의 디렉토리가 있을 때 덮어쓰지 않도록 설정
)

# Finish the W&B run
wandb.finish()

# python yolov8_train.py