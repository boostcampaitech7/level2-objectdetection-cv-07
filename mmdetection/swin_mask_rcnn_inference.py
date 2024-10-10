import os
import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
import csv
from pycocotools.coco import COCO
import numpy as np
from glob import glob
# 설정 및 체크포인트 파일 경로
config_file = '../../configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
checkpoint_file = './work_dirs/recycling_detection/latest.pth'
# 설정 파일 로드 및 수정
cfg = Config.fromfile(config_file)
num_classes = 10 # 학습 코드에서 설정한 클래스 수
cfg.model.roi_head.bbox_head.num_classes = num_classes
cfg.model.roi_head.mask_head.num_classes = num_classes
# 모델 로드
model = init_detector(cfg, checkpoint_file, device='cuda:0')
# COCO 포맷에서 클래스 이름 가져오기
coco = COCO('/data/ephemeral/home/data/dataset/test.json')
classes = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
# 테스트 이미지 목록
test_image_dir = '/data/ephemeral/home/data/dataset/test'
test_images = glob(os.path.join(test_image_dir, '*.jpg'))
print(f"Found {len(test_images)} test images.")
results = []
for image_path in test_images:
  # 추론 수행
  result = inference_detector(model, image_path)
  # 결과를 Pascal VOC 형식으로 변환
  for class_id, bboxes in enumerate(result):
    for bbox in bboxes:
      if bbox[4] >= 0.5: # 신뢰도 임계값
        x1, y1, x2, y2, score = bbox
        class_name = classes[class_id]
        results.append([
          os.path.basename(image_path), # 이미지 ID (파일명만 추출)
          score, # 신뢰도
          x1, y1, x2, y2, # 바운딩 박스 좌표
          class_id # 클래스 ID
        ])
# CSV 파일로 저장
with open('submission.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['image_id', 'PredictionString'])
  for result in results:
    image_id = result[0]
    pred_string = ' '.join(map(str, result[1:]))
    writer.writerow([image_id, pred_string])
print("Prediction completed and saved to submission.csv")