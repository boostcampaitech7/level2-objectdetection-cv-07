import os
import sys


# 상위 폴더를 sys.path에 추가
sys.path.append('/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/Co-DETR-main')


from mmcv import Config
from mmdet.utils import get_device
from projects import *



def get_current_filename():
    # __file__ 변수로 파일 경로를 얻고, os.path.basename()으로 파일명만 추출
    filename = os.path.basename(__file__)
    return filename

def get_config():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    #cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    #print(cfg.data.train)
    #print(cfg.model)
    #cfg = Config.fromfile('/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/mmdetection/ssd_experiment/ssd300_coco.py')
    cfg = Config.fromfile('/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/CoDetr/Co-Detr/model_configs/co_dino_5scale_r50_1x_coco.py')



    root='/data/ephemeral/home/dataset/'

    # dataset config 수정
    if 'dataset' in cfg.data.train:
        cfg.data.train.dataset.classes = classes
        cfg.data.val.dataset.classes = classes
        cfg.data.test.dataset.classes = classes
    else:
        cfg.data.train.classes = classes
        cfg.data.val.classes = classes
        cfg.data.test.classes = classes


    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보
    cfg.data.img_norm_cfg = dict(mean=[123.675, 116.28, 110.53], std=[60, 59, 61], to_rgb=True)

    #cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize faster-rcnn

    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    filename = get_current_filename()
    cfg.work_dir = f'/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/CoDetr/work_dirs/{filename}'


    cfg.model.bbox_head[0].num_classes = 10
    cfg.model.query_head.num_classes = 10
    cfg.model.roi_head[0].bbox_head.num_classes = 10

    #cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config = dict(grad_clip=None)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    return cfg