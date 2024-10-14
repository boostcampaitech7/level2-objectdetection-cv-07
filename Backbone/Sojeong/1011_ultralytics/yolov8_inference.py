import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from ultralytics import YOLO
import cv2

# YOLOv8 모델을 사용한 추론 코드
def main():
    # 경로 설정
    annotation = '/data/ephemeral/home/data/images/test.json'  # annotation 경로
    data_dir = '/data/ephemeral/home/data/images/test'  # 테스트 이미지 경로
    score_threshold = 0.05  # confidence threshold
    check_point = '/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/runs/detect/exp12/weights/best.pt'  # YOLOv8 체크포인트 경로
    
    # YOLOv8 모델 불러오기
    model = YOLO(check_point)
    
    # device 설정 (GPU 사용 여부)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # YOLOv8 inference (테스트 이미지에서 예측)
    results = model(data_dir, conf=score_threshold)  # 테스트 이미지에 대한 추론 결과
    
    prediction_strings = []
    file_names = []
    
    # COCO 데이터셋 불러오기
    coco = COCO(annotation)

    # submission 파일 생성
    for i, result in enumerate(results):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]

        # 결과에서 박스, 신뢰도, 라벨 추출
        for box, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if score > score_threshold:
                # YOLOv8에서의 label은 0부터 시작, score와 bounding box 좌표를 prediction string에 추가
                prediction_string += f"{int(label)} {score} {box[0]} {box[1]} {box[2]} {box[3]} "
                #  (xmin, ymin, xmax, ymax)
        prediction_strings.append(prediction_string.strip())  # trailing space 제거
        file_names.append(image_info['file_name'])

    # submission 데이터프레임 생성
    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })
    
    # 제출 파일로 저장
    submission.to_csv('./yolov8_submission.csv', index=False)
    print("Submission saved as yolov8_submission.csv")
    print(submission.head())


if __name__ == "__main__":
    main()

# python /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/1011_ultralytics/yolov8_inference.py