#클래스별 어노테이션 개수
import csv
from collections import defaultdict

# 클래스 정의 
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# 클래스별 어노테이션 개수를 저장할 딕셔너리
class_annotations = defaultdict(int)
total_annotations = 0 #총 어노테이션 개수

# CSV 파일 경로
csv_file = '/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/mmdetection/work_dirs/detr_r50/submission_latest.csv'  # CSV 파일의 경로

# CSV 파일 읽기
def count_annotations_per_class(file_path):
    global total_annotations  # 전역 변수로 선언하여 사용
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기

        # 각 행마다 클래스 ID에 따라 카운트 증가
        for row in reader:
            predictions = row[0].split()  # 공백으로 구분된 예측값들
            
            # 예측된 각 객체 처리
            for i in range(0, len(predictions), 6):
                class_id = int(float(predictions[i]))  # 첫 번째 요소는 class_id
                class_annotations[classes[class_id]] += 1
                total_annotations += 1 

    return class_annotations

# 결과 출력
if __name__ == "__main__":
    annotation_counts = count_annotations_per_class(csv_file)

    print("클래스별 어노테이션 개수:")
    for class_name, count in annotation_counts.items():
        print(f"{class_name}: {count}")
    print(f"\n총 어노테이션 개수: {total_annotations}")