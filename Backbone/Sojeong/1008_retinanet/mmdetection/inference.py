import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import os
import pandas as pd
from pycocotools.coco import COCO
import argparse

# 명령줄 인자 파서를 정의
def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for object detection")
    
    # 명령줄 인자 정의 (train.sh에서 넘겨받은 것)
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--root', type=str, required=True, help='dataset path')
    parser.add_argument('--num_classes', type=int, required=True, help='number of classes')

    return parser.parse_args()

def load_config(config_path, root, num_classes):
    """Load and update the configuration file."""
    cfg = Config.fromfile(config_path)

    # 테스트 데이터셋 관련 설정 업데이트
    cfg.data.test.classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
                             "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    #cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # 이미지 크기 조정
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.gpu_ids = [1]
    
    # 모델에 따라 roi_head 또는 bbox_head 설정
    if hasattr(cfg.model, 'roi_head'):  # Faster R-CNN 같은 모델의 경우
        cfg.model.roi_head.bbox_head.num_classes = num_classes
    elif hasattr(cfg.model, 'bbox_head'):  # RetinaNet 같은 모델의 경우
        cfg.model.bbox_head.num_classes = num_classes

    return cfg


def build_model_and_load_checkpoint(cfg, args, epoch='latest'):
    """학습된 체크포인트로부터 모델을 불러옴."""
    work_dir = f'./work_dirs/{args.model}'
    checkpoint_path = os.path.join(work_dir, f'{epoch}.pth')

    # 모델을 빌드하고 체크포인트 불러오기
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = cfg.data.test.classes

    # 모델 병렬화 설정
    model = MMDataParallel(model.cuda(), device_ids=[0])

    return model

def create_dataloader(cfg):
    """테스트 데이터셋과 dataloader를 생성."""
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    return dataset, data_loader

def generate_submission(output, cfg, args, class_num=10):
    """추론 결과를 후처리하여 submission.csv 파일 생성."""
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)

    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += f"{j} {o[4]} {o[0]} {o[1]} {o[2]} {o[3]} "

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    # submission 파일 생성 및 저장
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    work_dir = f'./work_dirs/{args.model}'
    submission_path = os.path.join(work_dir, f'submission_{args.model}.csv')
    submission.to_csv(submission_path, index=None)

    print(f"Submission saved at {submission_path}")
    return submission

def main():
    # 명령줄 인자 파싱
    args = parse_args()

    # 모델에 맞는 config 경로 설정
    #config_path = f"./configs/yolox/{args.model}.py"  
    #config_path = "./configs/retinanet/retinanet_r50_caffe_fpn_mstrain_1x_coco.py"
    config_path = "/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/autoassign/autoassign_r50_fpn_8x2_1x_coco.py"
    cfg = load_config(config_path, args.root, args.num_classes)

    # 모델을 빌드하고 체크포인트 불러오기
    model = build_model_and_load_checkpoint(cfg, args)

    # 데이터로더 생성
    dataset, data_loader = create_dataloader(cfg)

    # 싱글 GPU로 추론 실행
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # submission 파일 생성
    generate_submission(output, cfg, args, args.num_classes)

if __name__ == "__main__":
    main()
