import argparse
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_root_logger

def setup_config(config_file, root):
    cfg = Config.fromfile(config_file)
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = os.path.join(root, 'train.json')
    
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')

    return cfg

def build_and_train_model(cfg):
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)


def main():
    root = '/data/ephemeral/home/data/dataset/'
    cfg = setup_config('./configs/swin/faster_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py', root)

    cfg.data.samples_per_gpu = 4
    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = './work_dirs/faster_rcnn_swin-s_trash'
    cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.model.roi_head.mask_head = None
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    build_and_train_model(cfg)

if __name__ == '__main__':
    main()
