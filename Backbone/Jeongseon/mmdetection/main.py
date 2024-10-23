import sys

sys.path.append('/data/ephemeral/home/Jeongseon/mmdetection/base')

sys.path.append('/data/ephemeral/home/Jeongseon/mmdetection/wandb') ## wandb 디렉토리 경로 추가
from wandb_utils import init_wandb, setup_wandb_run, setup_wandb_logger

import os
from dotenv import load_dotenv
import wandb
import argparse
import yaml
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
from mmdet.utils import get_device

import pdb

#train_utils.py로 함수들 다 빼기
#변수업데이트 함수
#중첩된 딕셔너리에서 키를 찾아 값을 업데이트하는 함수
def update_nested_dict(d, key, value):
    if key in d:
        d[key] = value
        return True
    for v in d.values():
        if isinstance(v, dict):
            if update_nested_dict(v, key, value):
                return True
    return False

#설정을 업데이트하는 함수
def update_config(cfg, updates):
    for key, value in updates.items(): 
        if not update_nested_dict(cfg, key, value):
            print(f"Warning: Key '{key}' not found in config.")

#모델 타입에 따른 기본 설정 파일 경로를 반환
def get_model_config(model_type):
    config_paths = {
        'SwinTransformer' : '/data/ephemeral/home/Jeongseon/mmdetection/configs/swin/faster_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
        'EfficientNet_RetinaNet': '/data/ephemeral/home/Jeongseon/mmdetection/configs/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py',
        'DETR': 'configs/detr/detr_r50_8x2_150e_coco.py',
        'YOLOV3': '/data/ephemeral/home/Jeongseon/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
    }
    return config_paths.get(model_type)


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()

    # 실험 설정 로드
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # .env 파일에서 환경 변수 로드
    #load_dotenv()

    # wandb 자동 로그인 설정
    #os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
    #os.environ["WANDB_PROJECT"] = os.getenv('WANDB_PROJECT')

    # wandb 초기화
    #wandb.init()

    # 기본 설정 파일 로드
    base_config_path = get_model_config(exp_config['backbone'])
    cfg = Config.fromfile(base_config_path)

    # 실험 설정으로 cfg 업데이트
    updates = {
        'samples_per_gpu': exp_config['batch_size'],
        'lr': exp_config['learning_rate'],
        'weight_decay': exp_config['weight_decay'],
        'max_epochs': exp_config['num_epochs'],
        'num_classes': len(exp_config['classes'])
    }

    update_config(cfg, updates)

    # 클래스 정보 설정
    if 'dataset' in cfg.data.train:
        cfg.data.train.dataset.classes = exp_config['classes']
        cfg.data.val.dataset.classes = exp_config['classes']
        cfg.data.test.dataset.classes = exp_config['classes']
    else:
        cfg.data.train.classes = exp_config['classes']
        cfg.data.val.classes = exp_config['classes']
        cfg.data.test.classes = exp_config['classes']

    cfg.gpu_ids = [0]

    # 작업 디렉토리 설정
    if args.work_dir:
        cfg.work_dir = args.work_dir
    
    # 데이터셋 경로 설정
    data_root = '/data/ephemeral/home/data/dataset'
    cfg.data.train.ann_file = f'{data_root}/split/train_0_5.json'
    cfg.data.train.img_prefix = f'{data_root}/'
    cfg.data.val.ann_file = f'{data_root}/split/valid_0_5.json'
    cfg.data.val.img_prefix = f'{data_root}/'
    cfg.data.test.ann_file = f'{data_root}/test.json'
    cfg.data.test.img_prefix = f'{data_root}/'

    # 랜덤 시드 설정
    cfg.seed = 2022
    cfg.device = get_device()
    #cfg.data.workers_per_gpu = 0 #

    if 'roi_head' in cfg.model:
         # mask_head 부분을 제거하거나 다음과 같이 설정
        cfg.model.roi_head.mask_head = None
    else:
        pass

    # mask_head 부분을 제거하거나 다음과 같이 설정
    #cfg.model.roi_head.mask_head = None

    #pdb.set_trace()

    # 데이터셋 및 모델 빌드
    datasets = [build_dataset(cfg.data.train)]
    #print(cfg.data.train) 
    #print(datasets)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 학습 시작
    train_detector(model, datasets, cfg, distributed=False, validate=True) #distributed=False

if __name__ == '__main__':
    main()