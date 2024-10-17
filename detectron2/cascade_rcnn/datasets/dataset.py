import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

class register_datasets:
    # Register Dataset
    try:
        register_coco_instances('coco_trash_train', {}, '/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/result/train_0_10.json', '/data/ephemeral/home/data/dataset/')
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/result/valid_0_10.json', '/data/ephemeral/home/data/dataset/')
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]