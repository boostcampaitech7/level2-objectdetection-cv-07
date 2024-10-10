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
def get_cfg(config_path, classes, root, epoch='latest'):
    cfg = Config.fromfile(config_path)
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # Resize
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.seed = 2021
    cfg.gpu_ids = [1]
    cfg.work_dir = 'mmdetection/work_dirs/yolov3_trash'
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
def load_model(cfg, dataset, epoch='latest'):
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))  # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')  # load checkpoint
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    return model

# 함수: Inference 및 결과 저장
def inference(model, data_loader, cfg, epoch='latest'):
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # output 계산

    # COCO 형식의 데이터 로드
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    # 결과 처리
    prediction_strings = []
    file_names = []
    class_num = cfg.model.bbox_head.num_classes

    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += f'{j} {o[4]} {o[0]} {o[1]} {o[2]} {o[3]} '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })

    # 결과 저장
    submission_file = os.path.join(cfg.work_dir, f'submission_{epoch}.csv')
    submission.to_csv(submission_file, index=False)
    return submission_file

# 전체 inference 파이프라인 실행 함수
def run_inference(config_path, root, classes, epoch='latest'):
    cfg = get_cfg(config_path, classes, root, epoch)
    dataset, data_loader = build_data(cfg)
    model = load_model(cfg, dataset, epoch)
    result_file = inference(model, data_loader, cfg, epoch)
    print(f"Inference complete. Results saved to {result_file}")