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



# 원본 이미지 inference 저장

# 모델을 평가 모드로 설정
model.eval()

# 예측 수행
prediction_strings = []
file_names = []

for data in tqdm(test_loader):
    
    prediction_string = ''
    input=data[0]
    with torch.no_grad():
        outputs = model(data)[0]['instances']  # model에 올바른 형식으로 전달
    
    # 예측 결과 처리
    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()
    
    for target, box, score in zip(targets, boxes, scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
    
    prediction_strings.append(prediction_string)
    file_names.append(input['file_name'].replace('/data/ephemeral/home/data/dataset/', ''))

# 제출 파일 생성
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join('/data/ephemeral/home/Seungcheol/ouput2','submission_2_5.csv'), index=False)



