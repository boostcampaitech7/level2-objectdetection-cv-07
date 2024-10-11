import detectron2.utils.comm as comm
from detectron2.engine import HookBase
import wandb
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import torch

def visualize_prediction(image, predictions, cfg):
    # 예측 결과 시각화
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def log_prediction_to_wandb(image, predictions, step, cfg):
    # 예측 결과 시각화
    visualized_image = visualize_prediction(image, predictions, cfg)
    
    # 시각화된 이미지를 WandB에 기록
    wandb.log({
        "prediction": [wandb.Image(visualized_image, caption=f"Step {step}")]
    })

class WandbHook(HookBase):
    def __init__(self, cfg, val_loader):
        self.cfg = cfg
        self.val_loader = val_loader
        self.evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, True, output_dir="./output/")  # evaluator 초기화

    def after_step(self):
        # iteration이 끝날 때마다 한 번씩 검증
        if self.trainer.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and self.trainer.iter!=0:  # 매 100 iteration마다
            # mAP 평가 수행 및 기록
            results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
            mAP = results["bbox"]["AP"]  # COCOEvaluator 결과에서 mAP 추출

            if comm.is_main_process():
                wandb.log({
                    "mAP": mAP,
                    "loss": self.trainer.storage.latest()["total_loss"],
                    "lr": self.trainer.storage.latest()["lr"],
                    "iteration": self.trainer.iter
                })


            with torch.no_grad():
                for batch in self.val_loader:
                    # 모델 예측
                    image = batch[0]["image"].numpy().transpose(1, 2, 0)
                    outputs = self.trainer.model(batch)

                    # 시각화 및 WandB로 로그
                    log_prediction_to_wandb(image, outputs, self.trainer.iter, self.cfg)
                    break  # 한 번만 예측하고 나감