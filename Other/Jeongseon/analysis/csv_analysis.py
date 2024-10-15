import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import colorsys
#수정
# threshold 조정할 때 csv 분석 부분은 바뀌지 않음 -> 수정 (이제 바뀜)

# 클래스 정의
CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
BASE_PATH = '/data/ephemeral/home/data/dataset'

def load_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file, dtype={'PredictionString': str, 'image_id': str})
    return df

def parse_predictions(prediction_string, threshold=0.0):
    if pd.isna(prediction_string) or prediction_string == '':
        return []
    
    try:
        predictions = prediction_string.split()
        objects = []
        for i in range(0, len(predictions), 6):
            class_id = int(float(predictions[i]))
            confidence = float(predictions[i+1])
            x_min, y_min, x_max, y_max = map(float, predictions[i+2:i+6])
            
            # 신뢰도 임계값에 따른 필터링
            if confidence >= threshold:
                objects.append({
                    'class': CLASSES[class_id],  # 클래스 ID를 클래스 이름으로 변환
                    'confidence': confidence,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
        return objects
    except Exception as e:
        st.error(f"Error parsing prediction string: {prediction_string[:50]}...")
        st.error(f"Error message: {str(e)}")
        return []

def calculate_metrics(csv_data, threshold=0.0):
    metrics = {
        'confidence': [],
        'objects_per_image': [],
        'class_distribution': {class_name: 0 for class_name in CLASSES}
    }
    for _, row in csv_data.iterrows():
        objects = parse_predictions(row['PredictionString'], threshold)
        if objects:
            metrics['confidence'].extend([obj['confidence'] for obj in objects])
            metrics['objects_per_image'].append(len(objects))
            for obj in objects:
                metrics['class_distribution'][obj['class']] += 1
    
    return metrics

# 이미지 로드 함수
def load_image(image_path):
    return Image.open(image_path)

# 색상 생성 함수
def generate_colors(n):
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), [(i/n, 1., 1.) for i in range(n)]))

COLORS = generate_colors(len(CLASSES))

# Bounding box 그리기 함수
def draw_bounding_boxes(image, objects):
    draw = ImageDraw.Draw(image)
    font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
    font = ImageFont.truetype(font_path, 25)

    for obj in objects:
        bbox = obj['bbox']
        x_min, y_min, x_max, y_max = bbox
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        color = tuple(int(x * 255) for x in COLORS[CLASSES.index(obj['class'])]) 
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        draw.text((x_min, y_min-15), f"{obj['class']}: {obj['confidence']:.2f}", fill=color, font=font)
    return image

def main():
    st.title("Object Detection Analysis Dashboard")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        csv_data = load_csv_file(uploaded_file)
        st.success("파일 업로드 성공!")

        ###################### Threshold 슬라이더 추가 #############################
        threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.write(f"Selected threshold: {threshold}")

        ###################### 이미지 시각화 섹션#############################
        st.header("Image Visualization")
        
        # 이미지 선택
        image_ids = csv_data['image_id'].unique()
        selected_image_id = st.selectbox("이미지 선택 :", image_ids)

        
        if selected_image_id:
            # 이미지 경로 (실제 환경에 맞게 수정 필요)
            image_path = f"{BASE_PATH}/{selected_image_id}"
            
            if os.path.exists(image_path):
                image = load_image(image_path)
                row = csv_data[csv_data['image_id'] == selected_image_id].iloc[0]
                objects = parse_predictions(row['PredictionString'], threshold)
                
                image_with_boxes = draw_bounding_boxes(image.copy(), objects)
                
                # 이미지 표시
                st.image(image_with_boxes, caption=f"Predictions for {selected_image_id}", use_column_width=True)
                
                # 예측 결과 표시
                #st.subheader("Prediction Details")
                #for obj in objects:
                #    st.write(f"Class: {obj['class']}, Confidence: {obj['confidence']:.2f}")
                
                # 클래스별 예측 수 그래프
                st.subheader("클래스 당 예측 수")
                class_counts = pd.Series([obj['class'] for obj in objects]).value_counts()
                fig = px.bar(x=class_counts.index, y=class_counts.values,
                            labels={'x': 'Class', 'y': 'Count'},
                            title="클래스 당 예측 수")
                st.plotly_chart(fig, use_column_width=True)

            else:
                st.error(f"이미지 파일 찾지 못함: {image_path}")

        ######################csv 결과 분석 #############################       
        metrics = calculate_metrics(csv_data, threshold)

        st.header("CSV 파일 분석")

        # 신뢰도 분포
        fig_confidence = px.histogram(metrics['confidence'], nbins=50,
                                      labels={'value': 'Confidence', 'count': 'Frequency'},
                                      title="신뢰도 분포")
        st.plotly_chart(fig_confidence)

        # 이미지당 객체 수 분포
        fig_objects = px.histogram(metrics['objects_per_image'], nbins=30,
                                   labels={'value': 'Number of Objects', 'count': 'Frequency'},
                                   title="이미지당 객체 수 분포")
        st.plotly_chart(fig_objects)

        # 클래스 분포
        class_dist = pd.DataFrame(list(metrics['class_distribution'].items()), columns=['Class', 'Count'])
        fig_class = px.bar(class_dist, x='Class', y='Count', title="클래스 분포")
        fig_class.update_xaxes(tickangle=45)
        st.plotly_chart(fig_class)

        # 클래스별 평균 신뢰도
        class_confidences = {class_name: [] for class_name in CLASSES}
        for _, row in csv_data.iterrows():
            objects = parse_predictions(row['PredictionString'], threshold)
            for obj in objects:
                class_confidences[obj['class']].append(obj['confidence'])
        
        avg_confidences = {class_name: np.mean(confidences) if confidences else 0 
                           for class_name, confidences in class_confidences.items()}
        avg_conf_df = pd.DataFrame(list(avg_confidences.items()), columns=['Class', 'Average Confidence'])
        fig_avg_conf = px.bar(avg_conf_df, x='Class', y='Average Confidence', 
                              title="클래스별 평균 신뢰도")
        fig_avg_conf.update_xaxes(tickangle=45)
        st.plotly_chart(fig_avg_conf)

        # 요약 통계
        st.subheader("요약 통계")
        col1, col2, col3 = st.columns(3)
        col1.metric("평균 신뢰도", f"{np.mean(metrics['confidence']):.4f}")
        col2.metric("이미지당 평균 Object", f"{np.mean(metrics['objects_per_image']):.2f}")
        col3.metric("전체 어노테이션 수", sum(metrics['objects_per_image']))

        # 원본 데이터 표시
        st.subheader("원본 데이터 표시")
        st.dataframe(csv_data)

if __name__ == "__main__":
    main()
