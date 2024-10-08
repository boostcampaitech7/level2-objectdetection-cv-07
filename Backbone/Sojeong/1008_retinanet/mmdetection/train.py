# train.py
import argparse
import os 
import wandb
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device
from mmcv.runner import HOOKS, LoggerHook
from dotenv import load_dotenv  # dotenv로 환경 변수 로드

# 사용할 수 있는 모델과 그에 따른 config 파일 경로 설정
AVAILABLE_MODELS = {
    'faster_rcnn': './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    'mask_rcnn': './configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
    'retinanet': './configs/retinanet/retinanet_r50_fpn_1x_coco.py'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train model for object detection")

    # 모델 선택 인자 추가
    parser.add_argument('--model', type=str, default='faster_rcnn',
                        choices=AVAILABLE_MODELS.keys(),
                        help='사용할 모델을 선택하세요 (faster_rcnn, mask_rcnn, retinanet)')
    
    # 기존 인자들 추가
    parser.add_argument('--root', type=str, default='/data/ephemeral/home/data/dataset/',
                        help='Root directory for the dataset') 
    parser.add_argument('--classes', type=str, nargs='+',
                        default=["General trash", "Paper", "Paper pack", "Metal", "Glass", 
                                 "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"],
                        help='List of class names for the dataset')
    parser.add_argument('--img_scale', type=int, nargs=2, default=(512, 512),
                        help='Image scale for resizing (width, height)')
    parser.add_argument('--samples_per_gpu', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU ids to use')
    parser.add_argument('--work_dir', type=str, default=None,  # work_dir을 명시하지 않으면 자동 설정
                        help='Directory to save checkpoints (자동으로 model 이름에 맞게 설정됩니다)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for object detection')
    parser.add_argument('--grad_clip_max_norm', type=float, default=35,
                        help='Max norm for gradient clipping')
    parser.add_argument('--grad_clip_norm_type', type=int, default=2,
                        help='Norm type for gradient clipping') # 2: L2 norm
    parser.add_argument('--max_keep_ckpts', type=int, default=3,
                        help='Max number of checkpoints to keep')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Interval for saving checkpoints') # Save every epoch
    parser.add_argument('--wandb_project', type=str, default='faster-rcnn-training',
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default='faster-rcnn-run',
                        help='WandB run name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity or team name')

    return parser.parse_args()


def load_config(args): 
    config_path = AVAILABLE_MODELS[args.model]  # 선택한 모델에 따라 config 경로 설정
    return Config.fromfile(config_path)


def update_config(cfg, args):
    # work_dir을 모델 이름에 맞춰 동적으로 설정
    if args.work_dir is None:
        args.work_dir = f'./work_dirs/{args.model}'
    
    # Update dataset configs
    cfg.data.train.classes = args.classes
    cfg.data.train.img_prefix = args.root
    cfg.data.train.ann_file = args.root + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = tuple(args.img_scale)

    cfg.data.test.classes = args.classes
    cfg.data.test.img_prefix = args.root
    cfg.data.test.ann_file = args.root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = tuple(args.img_scale)

    # Update other configs
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir

    # 모델에 맞춰 num_classes 업데이트 (roi_head와 bbox_head 모두 있는 경우와 bbox_head만 있는 경우)
    if hasattr(cfg.model, 'roi_head'):
        cfg.model.roi_head.bbox_head.num_classes = args.num_classes
    elif hasattr(cfg.model, 'bbox_head'):
        cfg.model.bbox_head.num_classes = args.num_classes

    cfg.optimizer_config.grad_clip = dict(max_norm=args.grad_clip_max_norm, norm_type=args.grad_clip_norm_type)
    cfg.checkpoint_config = dict(max_keep_ckpts=args.max_keep_ckpts, interval=args.checkpoint_interval)
    cfg.device = get_device()



def build_train_dataset(cfg):
    return [build_dataset(cfg.data.train)]


def build_model(cfg):
    model = build_detector(cfg.model)
    model.init_weights()
    return model

def setup_wandb(cfg, args):
    wandb_project = args.wandb_project
    wandb_run_name =args.wandb_run_name
    wandb_entity = args.wandb_entity

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=cfg,
        entity=wandb_entity
    )


@HOOKS.register_module()
class WandBLoggerHook(LoggerHook):
    """
    W&B Logger Hook to log loss and other metrics after each iteration.
    """
    def log(self, runner):
        wandb.log({
            'loss': runner.outputs['loss'],
            'learning_rate': runner.current_lr()[0],
            'epoch': runner.epoch,
            'iter': runner.iter
        })

def train_model(model, datasets, cfg):
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)


def main():

    args = parse_args()

    cfg = load_config(args)  # 모델에 맞는 config 파일 로드
    update_config(cfg, args)
    
    setup_wandb(cfg, args)

    datasets = build_train_dataset(cfg)
    model = build_model(cfg)
    train_model(model, datasets, cfg)


if __name__ == "__main__":
    main()

