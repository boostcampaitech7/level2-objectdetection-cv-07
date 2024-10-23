import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from utils.mapper import MyMapper
from wandbs.wandbhook import WandbHook
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog


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
    
    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])  # Validation 데이터 로더
        dataset_dicts = DatasetCatalog.get(self.cfg.DATASETS.TEST[0])  # 데이터셋 딕셔너리 가져오기
        hooks.append(WandbHook(cfg=self.cfg, val_loader=val_loader, dataset_dicts=dataset_dicts))
        return hooks
