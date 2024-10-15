import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO

# 함수: Config 파일 설정
def get_cfg(config_file_path, work_dir, dataset_root, classes, epoch):
    cfg = Config.fromfile(config_path)
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = dataset_root
    cfg.data.test.ann_file = f'{dataset_root}/test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # Resize
    
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.seed = 2021
    cfg.gpu_ids = [1]
    cfg.work_dir = work_dir
    cfg.model.bbox_head.num_classes = len(classes)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    return cfg

# 함수: 데이터셋 및 데이터로더 생성
def build_data(cfg):
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    return dataset, data_loader

# 함수: 모델 및 체크포인트 불러오기
def load_model(cfg, dataset, epoch):
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))  # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')  # load checkpoint
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    return model

# 함수: Inference 및 결과 저장
def inference(model, data_loader, cfg, epoch):
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # output 계산

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })
    submission_file = os.path.join(cfg.work_dir, f'submission_{epoch}.csv')
    submission.to_csv(submission_file, index=False)
    return submission