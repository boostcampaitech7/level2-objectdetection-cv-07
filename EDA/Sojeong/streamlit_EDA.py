import streamlit as st
import os
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import pandas as pd
import shutil
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.set_page_config(page_title="Object Detection Data", layout="centered")

# 스타일 적용
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .stSelectbox, .stTextInput {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit의 session state 초기화
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = 'train'
if 'category_filter' not in st.session_state:
    st.session_state.category_filter = "All"

# 데이터셋 경로 설정
dataset_path = '/data/ephemeral/home/data/dataset'

# 클래스 이름 정의 (영어)
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# 페이지 제목
st.title("Object Detection Data")

# JSON 파일 로드
annotations_file = os.path.join(dataset_path, "train.json")
with open(annotations_file, 'r') as f:
    dataset = json.load(f)

# 이미지 정보와 어노테이션을 JSON 파일에서 분리
images_info = dataset["images"]
annotations_info = dataset.get("annotations", [])

# 현재 이미지 선택
image_filenames = [img['file_name'] for img in images_info]
current_image_file = image_filenames[st.session_state.current_image_index]
image_selection = st.selectbox("Select an image:", image_filenames, index=st.session_state.current_image_index)

# 이미지 선택 변경 시 인덱스 업데이트
if image_selection != current_image_file:
    st.session_state.current_image_index = image_filenames.index(image_selection)
    # 키를 변경하여 자동으로 재렌더링 유도
    st.session_state.changed = not st.session_state.get('changed', False)

# 현재 이미지 메타 데이터 가져오기
current_image_info = images_info[st.session_state.current_image_index]
image_id = current_image_info['id']
image_width = current_image_info['width']
image_height = current_image_info['height']
image_date = current_image_info['date_captured']

# 이미지 메타 정보 표시
st.markdown("### Image Metadata")
st.write(f"**File Name**: {current_image_info['file_name']}")
st.write(f"**Image Size**: {image_width} x {image_height}")

# 어노테이션 정보 필터링 (현재 이미지에 해당하는 어노테이션만)
current_annotations = [ann for ann in annotations_info if ann['image_id'] == image_id]

# 카테고리 필터 선택
category_options = ["All"] + [classes[ann['category_id']] for ann in current_annotations]
category_filter = st.selectbox("Category Filter:", category_options, key="category_filter")

# 어노테이션 시각화 (PIL)
current_image_path = os.path.join(dataset_path, current_image_file)
image = Image.open(current_image_path)

if current_annotations:
    draw = ImageDraw.Draw(image)
    font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
    font = ImageFont.truetype(font_path, 20)

    # 어노테이션 그리기 (필터링된 카테고리 적용)
    for annotation in current_annotations:
        category_name = classes[annotation['category_id']]
        if category_filter == "All" or category_name == category_filter:
            bbox = annotation['bbox']  # 바운딩 박스 정보 [x, y, width, height]

            # 바운딩 박스 그리기
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline="blue", width=2)

            # 레이블 그리기
            text_size = draw.textbbox((bbox[0], bbox[1] - 25), category_name, font=font)
            draw.rectangle(text_size, fill="blue")
            draw.text((bbox[0], bbox[1] - 25), category_name, fill="yellow", font=font)

# 이미지 출력 (가운데 정렬)
st.markdown("### Image with Annotations")
centered_image_col = st.columns([1, 3, 1])  # 왼쪽과 오른쪽 여백을 주어 가운데 정렬
with centered_image_col[1]:
    st.image(image, caption=f'Current Image: {current_image_file}', width=400)
    
# 카테고리별 어노테이션 개수 집계 및 시각화
st.markdown("#### Number of Annotations per Category")
categories = [classes[annotation['category_id']] for annotation in current_annotations if category_filter == "All" or classes[annotation['category_id']] == category_filter]
category_counts = pd.Series(categories).value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

# 테이블 및 시각화 표시
st.table(category_counts)

if not category_counts.empty:
    fig, ax = plt.subplots(1, 2, figsize=(6, 2))

    # 바 차트
    category_counts.set_index('Category').plot(kind='bar', ax=ax[0], legend=False)
    ax[0].set_title('Number of Annotations per Category', fontsize=8)  # 제목 폰트 크기 조정
    ax[0].set_ylabel('Count', fontsize=6)  # y축 레이블 폰트 크기 조정
    ax[0].set_xlabel('Category', fontsize=6)  # x축 레이블 폰트 크기 조정
    ax[0].tick_params(axis='x', labelsize=6)  # x축 눈금 폰트 크기 조정
    ax[0].tick_params(axis='y', labelsize=6)  # y축 눈금 폰트 크기 조정


    # 파이 차트
    category_counts.set_index('Category').plot(kind='pie', y='Count', autopct='%1.1f%%', ax=ax[1], legend=False, fontsize=6)  # 파이 차트 폰트 크기 조정
    ax[1].set_ylabel('')

    st.pyplot(fig)
    
# 어노테이션 정보 토글
st.markdown("#### Annotations Info")
with st.expander("Annotations INFO"):
    if current_annotations:
        st.json(current_annotations)
    else:
        st.write("No annotations available for this image.")

# streamlit run streamlit_EDA.py