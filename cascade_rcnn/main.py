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
from datasets.dataset import register_datasets
from models.trainer import MyTrainer

def setup_config(config_path):
    cfg = get_cfg()

    #cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.merge_from_file('config/config.yaml')
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    cfg.OUTPUT_DIR = './output'

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.MASK_ON=False


    cfg.TEST.EVAL_PERIOD = 3000

    return cfg

if __name__ == "__main__":
    config_path = "Misc/config.yaml"

    # 데이터셋 등록
    register_datasets()

    # Config 설정 및 모델 훈련
    cfg = setup_config(config_path)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
