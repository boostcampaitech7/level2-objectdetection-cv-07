import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from utils.mapper import MyMapper

class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyMapper, sampler=sampler)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
