import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import pandas as pd

# 모델 로드
model_path = '/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/runs/detect/exp12/weights/best.pt'  # 학습한 모델의 경로
model = YOLO(model_path)

# 클래스 ID와 클래스 이름을 매핑하는 딕셔너리
class_names = {
    0: "General trash",
    1: "Paper",
    2: "Paper pack",
    3: "Metal",
    4: "Glass",
    5: "Plastic",
    6: "Styrofoam",
    7: "Plastic bag",
    8: "Battery",
    9: "Clothing"
}

# 클래스 ID와 색상을 매핑하는 딕셔너리
class_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple",
    5: "orange",
    6: "pink",
    7: "cyan",
    8: "magenta",
    9: "lime"
}

# 중앙에 정렬된 헤더
st.markdown("<h1 style='text-align: center;'>재활용 품목 분류</h1>", unsafe_allow_html=True)

# 이미지 업로드
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    img = Image.open(uploaded_file)
    
    # 이미지 표시
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # 이미지에서 객체 탐지 수행
    results = model(img)

    # 신뢰도 임계값 설정
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

    # 결과 상세 정보 출력
    st.write("Detection Results:")

    # 객체의 경계 상자 좌표 및 클래스 출력
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 경계 상자 좌표를 CPU로 이동 후 NumPy 배열로 변환
        classes = results[0].boxes.cls.cpu().numpy()  # 클래스 정보를 CPU로 이동 후 NumPy 배열로 변환
        confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도 정보를 CPU로 이동 후 NumPy 배열로 변환

        # 신뢰도 임계값을 기준으로 결과 필터링
        mask = confidences >= confidence_threshold
        boxes = boxes[mask]
        classes = classes[mask]
        confidences = confidences[mask]

        # 클래스 ID를 클래스 이름으로 변환
        class_names_list = [class_names[int(cls)] for cls in classes]

        # 필터링된 결과를 이미지에 다시 그리기
        draw = ImageDraw.Draw(img) 
        font = ImageFont.load_default()  # 기본 폰트 사용
        
        for box, cls, conf in zip(boxes, classes, confidences):
            color = class_colors[int(cls)]  # 클래스 ID에 따른 색상 선택
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
            text = f"{class_names[int(cls)]} {conf:.2f}"
            text_size = draw.textsize(text, font=font)
            text_location = [box[0], box[1] - text_size[1]]
            textbox_location = [box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]]
            draw.rectangle(textbox_location, fill=color)
            draw.text(text_location, text, fill="white", font=font)  # 경계 상자 위에 클래스 이름과 신뢰도 추가

        # 필터링된 결과 이미지 출력
        st.image(img, caption='Filtered Detected Image', use_column_width=True)

        # DataFrame으로 변환하여 출력
        df = pd.DataFrame({
            'xmin': boxes[:, 0],
            'ymin': boxes[:, 1],
            'xmax': boxes[:, 2],
            'ymax': boxes[:, 3],
            'class': class_names_list,
            'confidence': confidences
        })
        st.write(df)
    else:
        st.write("No detections found.")
        
# streamlit run /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Other/Sojeong/demo_yolo.py