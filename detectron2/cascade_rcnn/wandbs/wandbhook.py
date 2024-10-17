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
    # 예측 결과가 리스트인지 확인하고 첫 번째 요소로 접근
    if isinstance(predictions, list):
        predictions = predictions[0]  # 첫 번째 배치의 예측값을 사용

    # 예측 결과 시각화
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def log_prediction_to_wandb(image, predictions,annotations ,step, cfg):
    # 예측 결과 시각화
    visualized_image = visualize_prediction(image, predictions, cfg)

    # 정답 시각화
    gt_image = visualize_ground_truth(image, annotations, cfg)
    
    # 시각화된 이미지를 WandB에 기록
    wandb.log({
        "comparison": [
            wandb.Image(visualized_image, caption=f"Prediction (Step {step})"),
            wandb.Image(gt_image, caption="ground truth")
        ]
    })

# 정답 시각화
def visualize_ground_truth(image, annotations, cfg):
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # annotations 리스트를 데이터셋 딕셔너리로 변환
    dataset_dict = {"annotations": annotations}
    out_gt = v.draw_dataset_dict(dataset_dict)
    
    return out_gt.get_image()[:, :, ::-1]




class WandbHook(HookBase):
    def __init__(self, cfg, val_loader, dataset_dicts):
        self.cfg = cfg
        self.val_loader = val_loader
        self.dataset_dicts = dataset_dicts  # 데이터셋 딕셔너리(정답 포함)를 전달

    def after_step(self):
        # iteration이 체크포인트 주기마다 평가가 이루어진 후 평가 결과를 가져오기
        if self.trainer.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and self.trainer.iter!=0:
            # storage에서 평가 결과 가져오기
            latest_eval_results = self.trainer.storage.latest()
            print(latest_eval_results)

            if comm.is_main_process():  # 메인 프로세스에서만 로그 기록
                wandb.log({
                    "AP" : latest_eval_results['bbox/AP'][0],
                    "loss": latest_eval_results["total_loss"][0],  # 손실
                    "lr": latest_eval_results["lr"][0],  # 학습률
                    "iteration": self.trainer.iter  # 현재 iteration
                })
            
            self.trainer.model.eval()

            with torch.no_grad():
                for batch in self.val_loader:
                    # 모델 예측
                    image = batch[0]["image"].numpy().transpose(1, 2, 0)
                    outputs = self.trainer.model(batch)

                    # dataset_dicts에서 정답(annotations) 정보 가져오기
                    annotations = self.dataset_dicts[0]["annotations"]

                    # 시각화 및 WandB로 로그
                    log_prediction_to_wandb(image, outputs, annotations,self.trainer.iter, self.cfg)
                    break  # 한 번만 예측하고 나감

            self.trainer.model.train()