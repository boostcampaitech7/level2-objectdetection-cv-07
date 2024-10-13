import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import pdb

#csv 이미지 시각화

# 설정
test_images_dir = '/data/ephemeral/home/dataset/'
csv_file = '/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/mmdetection/work_dirs/detr_r50/submission_latest.csv'  # CSV 경로
output_dir = 'output_visualizations/swin_t'  # 결과 이미지를 저장할 디렉토리

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 클래스 정의 (순서가 중요합니다!)
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# 클래스별 색상 설정
def generate_colors(num_classes):
    np.random.seed(42)  # 일관된 색상을 위해 시드 설정
    colors = [(int(r*255), int(g*255), int(b*255)) 
              for r, g, b in np.random.rand(num_classes, 3)]
    return colors

colors = generate_colors(len(classes))

# CSV 파일 읽기
def read_csv(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            img_name = row[-1]
            predictions = row[0].split()  # 공백으로 구분된 예측값들
            data[img_name] = predictions
    return data

# 예측 문자열 해석
def parse_predictions(predictions):
    objects = []
    for i in range(0, len(predictions), 6):
        class_id = int(float(predictions[i]))
        confidence = float(predictions[i+1])
        x, y, w, h = map(float, predictions[i+2:i+6])
        objects.append({
            'class': classes[class_id],
            'confidence': confidence,
            'bbox': (x, y, w, h)
        })
    return objects

# 객체 검출 결과 시각화
def visualize_detection(image_path, objects):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    for obj in objects:
        class_name = obj['class']
        confidence = obj['confidence']
        x, y, w, h = obj['bbox']

        # 좌표를 픽셀 값으로 변환
        left = int(x)
        top = int(y)
        right = int(x + w)
        bottom = int(y + h)

        color = colors[classes.index(class_name)]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# 메인 실행 부분
def main():
    print("CSV 파일 읽는 중...")
    detection_data = read_csv(csv_file)

    print("이미지 처리 중...")
    for img_name, predictions in tqdm(detection_data.items()):
        #pdb.set_trace() #디버깅
        img_path = os.path.join(test_images_dir, img_name)
        if os.path.exists(img_path):
            objects = parse_predictions(predictions)
            output_image = visualize_detection(img_path, objects)
            output_filename = f'vis_{os.path.basename(img_name)}'            
            output_path = os.path.join(output_dir, output_filename) #f'vis{img_name}' :os.path.basename을 하지 않았던 것이 오류?

            cv2.imwrite(output_path, output_image)
        else:
            print(f"경고: 이미지를 찾을 수 없습니다 - {img_path}")

    print(f"처리 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()