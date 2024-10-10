import mmcv
#from mmcv import Config
from mmengine.config import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.runner import EvalHook, CheckpointHook, WandbLoggerHook
import wandb
# 설정 파일 로드
cfg = Config.fromfile('./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py')
# WandB 초기화
wandb.init(project="object-detection", name="js")
# WandB 로거 설정 추가
cfg.log_config.hooks.append(
  dict(type='WandbLoggerHook',
     init_kwargs={'project': "object-detection"},
     interval=100,
     log_checkpoint=True,
     log_checkpoint_metadata=True,
     num_eval_images=100))

# 평가 및 체크포인트 설정 수정
cfg.evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', rule='greater')
cfg.checkpoint_config = dict(interval=1, save_optimizer=True, by_epoch=True, max_keep_ckpts=3)

# EvalHook과 CheckpointHook을 명시적으로 설정
#custom_hooks = [
#  EvalHook(interval=1, metric='bbox', save_best='bbox_mAP', rule='greater'),
#  CheckpointHook(interval=1, save_optimizer=True, by_epoch=True, max_keep_ckpts=3)
#]

# 학습 설정에 custom_hooks 추가
#cfg.custom_hooks = custom_hooks
# 학습 결과와 체크포인트 저장할 경로
cfg.work_dir = './work_dirs/recycling_detection'
# 데이터셋 설정 수정
cfg.dataset_type = 'CocoDataset'
cfg.data.train.type = 'CocoDataset'
cfg.data.train.ann_file = '/data/ephemeral/home/Jihwan/code/dataset/train.json'
cfg.data.train.img_prefix = '/data/ephemeral/home/Jihwan/code/dataset/train'
# 테스트 데이터셋 설정 (필요한 경우)
cfg.data.test.type = 'CocoDataset'
cfg.data.test.ann_file = '/data/ephemeral/home/Jihwan/code/dataset/test.json'
cfg.data.test.img_prefix = '/data/ephemeral/home/Jihwan/code/dataset/test'
# 클래스 수 설정 (쓰레기 품목 수에 맞게 조정)
cfg.model.roi_head.bbox_head.num_classes = 10 # 예시: 10개 클래스
cfg.model.roi_head.mask_head.num_classes = 10
# 학습 설정
cfg.optimizer.lr = 0.02 / 8 # 학습률 조정
cfg.lr_config.warmup = None

cfg.log_config.interval = 10
#에폭 수 조정
cfg.runner.max_epochs = 12
# 데이터셋 구축 (학습 데이터에서 검증 데이터 분할)
train_dataset = build_dataset(cfg.data.train)
#train_dataset, val_dataset = train_dataset.split(0.8) # 80%는 학습, 20%는 검증
# 모델 구축
model = build_detector(cfg.model)
model.init_weights()
# 학습 시작
train_detector(model, [train_dataset], cfg, distributed=False, validate=True)
# WandB 실행 종료
wandb.finish()