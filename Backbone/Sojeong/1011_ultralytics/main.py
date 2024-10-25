import os
import torch
import pandas as pd
import argparse
from ultralytics import YOLO
from pycocotools.coco import COCO
from utils.utils import load_config
import timm
from torch import nn

class SwinBackbone(nn.Module):
    def __init__(self):
        super(SwinBackbone, self).__init__()
        # Load the Swin Transformer Base model
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        
        # Remove the classification head and keep it as a feature extractor
        self.model.head = nn.Identity()

    def forward(self, x):
        # Forward pass through the Swin Transformer
        x = self.model(x)
        return x        

def train_model(config):
    # Load YOLO model
    model = YOLO(config["model_path"])

    # Swin base backbone
    model.model.backbone = SwinBackbone()
    
    # Train the model
    model.train(
        data=config["data"],  # path to dataset YAML
        epochs=config["epochs"],  # number of training epochs
        batch=config["batch_size"],  # batch size
        patience=config["patience"],  # early stopping patience
        imgsz=config["imgsz"],  # image size
        device=config["device"],  # device ID (GPU or CPU)
        amp=config["amp"],  # automatic mixed precision
        name=config["run_name"],  # run name
        exist_ok=False,  # prevent overwriting
        save_dir=config["save_dir"]  # save results to this directory
    )
    print("Training complete.")

def inference(config):
    # Load YOLO model with the best weights
    model = YOLO(config["checkpoint_path"])

    # Set device (GPU if available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Perform inference on the test images
    results = model(config["data_dir"], conf=config["score_threshold"], iou=config["iou_threshold"])

    prediction_strings = []
    file_names = []

    # Load COCO annotations
    coco = COCO(config["annotation_path"])

    # Process each result
    for i, result in enumerate(results):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]

        for box, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if score > config["score_threshold"]:
                prediction_string += f"{int(label)} {score} {box[0]} {box[1]} {box[2]} {box[3]} "

        prediction_strings.append(prediction_string.strip())
        file_names.append(image_info['file_name'])

    # Create submission dataframe
    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })

    # Save submission to CSV
    submission.to_csv(config["submission_save_path"], index=False)
    print(f"Submission saved as {config['submission_save_path']}")
    print(submission.head())

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="YOLOv8 Train and Inference")
    parser.add_argument('--mode', choices=['train', 'inference'], required=True, help="Mode to run: 'train' or 'inference'")
    args = parser.parse_args()

    # Load configuration
    config = load_config("config.json")

    # Run based on the mode
    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'inference':
        inference(config)
