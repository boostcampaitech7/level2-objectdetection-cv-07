import os
import copy
import torch
from tqdm import tqdm
import pandas as pd
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()
import torch

import json
import numpy as np
from detectron2.structures import Boxes, BoxMode, Instances


from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import albumentations as A


# Register Dataset
try:
    register_coco_instances('coco_trash_final', {}, '/data/ephemeral/home/data/dataset/test.json', '/data/ephemeral/home/data/dataset/')
except AssertionError:
    pass


MetadataCatalog.get('coco_trash_final').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                        "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]



# 설정 파일 로드 (LazyConfig)
cfg = LazyConfig.load('/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/detectron2/mvitv2_rcnn_cascade/MViTv2/configs/cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x.py')
cfg.dataloader.test.dataset.names = 'coco_trash_final'
cfg.dataloader.evaluator.dataset_name='coco_trash_final'
cfg.dataloader.test.batch_size = 1

model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
# 모델 가중치 로드
DetectionCheckpointer(model).load('/data/ephemeral/home/Seungcheol/ouput2/model_0006999.pth')

# 테스트 데이터셋 로드
test_loader = instantiate(cfg.dataloader.test)
test_loader.num_workers = 4 



# flip된 이미지 inference 후 박스 조정 후 저장

def flip_boxes_horizontally(boxes, image_width):
    """
    좌우 반전된 박스 좌표를 원본 이미지 좌표계로 변환합니다.
    
    Args:
        boxes (list of list or array): [x_min, y_min, x_max, y_max] 형식의 박스 리스트
        image_width (int): 이미지의 너비
    
    Returns:
        list of list: 변환된 박스 리스트
    """
    flipped_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        flipped_x_min = image_width - x_max
        flipped_x_max = image_width - x_min
        # 좌표가 음수가 되지 않도록 보정
        flipped_x_min = max(flipped_x_min, 0)
        flipped_x_max = min(flipped_x_max, image_width)
        flipped_boxes.append([flipped_x_min, y_min, flipped_x_max, y_max])
    return flipped_boxes

# 좌우 반전 변환기 정의
transform = A.HorizontalFlip(p=1.0)

# 예측 수행
prediction_strings = []
file_names = []
model.eval()

for data in tqdm(test_loader):
    prediction_string = ''
    input_data = data[0]
    image = input_data['image'].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    
    # 좌우 반전 적용
    transformed = transform(image=image)
    flipped_image = transformed['image']
    
    # 텐서로 변환
    flipped_image_tensor = torch.from_numpy(flipped_image).permute(2, 0, 1).float()  # Ensure dtype is float
    input_data['image'] = flipped_image_tensor
    
    # 모델 추론
    with torch.no_grad():
        outputs = model([input_data])[0]['instances']  # 모델 입력을 리스트로 감싸기
    
    # 예측 결과 처리 (좌우 반전된 박스의 좌표를 원래 이미지로 조정)
    image_width = image.shape[1]  # 이미지 너비 계산 (H, W, C) 형식
    targets = outputs.pred_classes.cpu().tolist()
    boxes = outputs.pred_boxes.tensor.cpu().numpy().tolist()
    scores = outputs.scores.cpu().tolist()
    
    # 박스 좌표 변환
    flipped_boxes = flip_boxes_horizontally(boxes, 1024)
    
    # 변환된 박스를 사용하여 예측 문자열 생성
    for target, box, score in zip(targets, flipped_boxes, scores):
        prediction_string += f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} "
    
    prediction_strings.append(prediction_string.strip())  # 마지막 공백 제거
    file_names.append(input_data['file_name'].replace('/data/ephemeral/home/data/dataset/', ''))

# 제출 파일 생성
submission = pd.DataFrame({
    'image_id': file_names,
    'PredictionString': prediction_strings
})
submission.to_csv(os.path.join('/data/ephemeral/home/Seungcheol/ouput2', 'submission_2_5_flip.csv'), index=False)