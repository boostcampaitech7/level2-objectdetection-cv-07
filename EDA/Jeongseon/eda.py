import streamlit as st
import os
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.colors

# 페이지 설정을 넓게
#st.set_page_config(layout="wide")

# 사이드바 생성
st.sidebar.title("Object Detection EDA")
st.sidebar.header("분석 모드")
analysis_mode = st.sidebar.radio(
    "선택:",
    ("개별 이미지 분석", "EDA"),
    index=0  # 기본 선택을 '개별 이미지 분석'으로 설정
)

# 메인 콘텐츠 영역
st.title("Object Detection EDA")

# 기본 데이터셋 경로 설정
base_path = '/data/ephemeral/home/data/dataset'

# COCO annotation 파일 로드
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# 색상 생성 함수
def generate_colors(n):
    hsv_tuples = [(x / n, 1., 1.) for x in range(n)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

# 바운딩 박스와 클래스 레이블 그리기 함수
def draw_bounding_boxes(image, annotations, image_id, categories, category_colors, selected_category='All'):
    draw = ImageDraw.Draw(image)
    font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
    font = ImageFont.truetype(font_path, 25)

    for ann in annotations:
        if ann['image_id'] == image_id:
            category = next(cat for cat in categories if cat['id'] == ann['category_id'])
            if selected_category == 'All' or category['name'] == selected_category:
                bbox = ann['bbox']
                label = category['name']
                color = tuple(int(x * 255) for x in category_colors[category['id']])
                
                # 바운딩 박스 그리기
                draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline=color, width=2)
                
                # 레이블 배경 그리기
                text_bbox = font.getbbox(label)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + text_height], fill=color)
                
                # 레이블 텍스트 그리기
                draw.text((bbox[0], bbox[1]), label, fill="white", font=font)

    return image


# 어노테이션 개수 그래프 생성 함수
def create_annotation_count_graph(annotations, categories, category_colors):
    category_counts = {cat['name']: 0 for cat in categories}
    for ann in annotations:
        category = next(cat for cat in categories if cat['id'] == ann['category_id'])
        category_counts[category['name']] += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    categories_names = list(category_counts.keys())
    counts = list(category_counts.values())
    
    colors = [category_colors[next(cat['id'] for cat in categories if cat['name'] == name)] for name in categories_names]
    
    bars = ax.bar(categories_names, counts, color=colors)
    ax.set_ylabel('Count the annotations')
    ax.set_title('Number of annotations per category')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')

    # 그래프 테두리 제거
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 점선 그리드 추가
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        if height > 0 :
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')

    # x축 라벨 위치 조정
    ax.tick_params(axis='x', which='major', pad=5)
    
    plt.tight_layout()
    
    return fig

# 분석 모드에 따른 내용 표시
if analysis_mode == "개별 이미지 분석":
    st.header("개별 이미지 분석")
    
    # 폴더 선택 (train / test)
    option = st.selectbox('폴더 선택', ('train', 'test'))

    # 선택된 폴더 경로와 annotation 파일 경로
    folder_path = os.path.join(base_path, option)
    annotation_path = os.path.join(base_path, f'{option}.json')

    # annotations 로드
    annotations = load_annotations(annotation_path)

    # 카테고리별 색상 생성 (0-1 범위로 유지)
    colors = generate_colors(len(annotations['categories']))
    category_colors = {cat['id']: color for cat, color in zip(annotations['categories'], colors)}

    # 이미지 파일 목록 가져오기 및 경로 처리
    image_files = [img['file_name'] for img in annotations['images']]
    image_files = [os.path.basename(f) if f.startswith(option) else f for f in image_files]
    image_files.sort()

    # 현재 이미지 인덱스를 세션 상태로 관리
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0

    # 이전/다음 버튼 기능
    def prev_image():
        st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(image_files)

    def next_image():
        st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(image_files)

    # 이미지 선택 함수
    def on_image_select():
        st.session_state.current_image_index = image_files.index(st.session_state.selected_image)

    # 이미지 선택
    selected_image = st.selectbox('이미지 선택', image_files, 
                                  index=st.session_state.current_image_index, 
                                  key='selected_image',
                                  on_change=on_image_select)

    # 이전/다음 버튼
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("이전", on_click=prev_image)
    with col3:
        st.button("다음", on_click=next_image)

    # 바운딩 박스 표시 여부
    show_bbox = st.checkbox("바운딩 박스 표시")

    # 현재 선택된 이미지
    current_image = image_files[st.session_state.current_image_index]

    # 선택된 이미지 표시
    if current_image:
        image_path = os.path.join(folder_path, current_image)
        try:
            image = Image.open(image_path)
            
            # 이미지 ID 찾기
            image_id = next(img['id'] for img in annotations['images'] if os.path.basename(img['file_name']) == current_image)

            # 현재 이미지의 어노테이션 가져오기
            current_image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            
            # 현재 이미지에 존재하는 클래스 찾기
            category_names = {cat['id']: cat['name'] for cat in annotations['categories']}
            existing_categories = set(category_names[ann['category_id']] for ann in current_image_annotations) #현재 이미지에 존재하는 클래스만
            
            # 클래스 필터링을 위한 selectbox 추가
            category_options = ['All'] + list(existing_categories)
            selected_category = st.selectbox('클래스 선택', category_options)
            
            if show_bbox:
                image_with_bbox = draw_bounding_boxes(image.copy(), current_image_annotations, image_id, annotations['categories'], category_colors, selected_category)
                st.image(image_with_bbox, caption=current_image, use_column_width=True)
            else:
                st.image(image, caption=current_image, use_column_width=True)
            
            st.success(f"이미지 경로: {image_path}")
            
            # 이미지 정보 표시
            st.write(f"파일 크기: {os.path.getsize(image_path)} bytes")
            st.write(f"이미지 크기: {image.size[0]} x {image.size[1]} 픽셀")

            # 바운딩 박스가 체크되었을 때만 그래프 표시
            if show_bbox:
                st.subheader("카테고리별 어노테이션 개수")

                # 현재 이미지의 어노테이션 개수 그래프 표시 (필터링 적용)
                current_image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
                if selected_category != 'All':
                    current_image_annotations = [ann for ann in current_image_annotations if category_names[ann['category_id']] == selected_category]
                fig = create_annotation_count_graph(current_image_annotations, annotations['categories'], category_colors)
                st.pyplot(fig)

        except IOError as e:
            st.error(f"이미지를 열 수 없습니다: {image_path}")
            st.error(f"에러 메시지: {str(e)}")
    else:
        st.info("이미지를 선택해주세요.")

elif analysis_mode == "EDA":
    st.header("EDA 기능 (Train 데이터셋)")
    
    # Train annotation 파일 경로
    train_annotation_path = os.path.join(base_path, 'train.json')
    
    # Train annotations 로드
    train_annotations = load_annotations(train_annotation_path)
    
    # 데이터프레임 생성
    df = pd.DataFrame(train_annotations['annotations'])
    
    # 1. 이미지당 어노테이션 수 분포
    st.subheader("1. 이미지당 어노테이션 수 분포")

    st.text(": 한 이미지에 몇 개의 Bounding box가 있는지, 그 분포를 나타낸 결과")
    
    annotation_counts = df['image_id'].value_counts()
    fig1 = px.histogram(annotation_counts, 
                        log_y=True, color_discrete_sequence=['indianred'], opacity=0.7,
                        labels={"value":"Number of Annotations Per Image"},
                        title="<b>DISTRIBUTION OF # OF ANNOTATIONS PER IMAGE   " +
                              "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
                        )
    fig1.update_layout(showlegend=False,
                       xaxis_title="<b>Number of Annotations Per Image</b>",
                       yaxis_title="<b>Count of Images</b>",)
    
    st.plotly_chart(fig1, use_container_width=True)
    st.write(f"총 이미지 수: {len(df['image_id'].unique())}")
    st.write(f"총 어노테이션 수: {len(df)}")
    st.write(f"이미지당 평균 어노테이션 수: {len(df) / len(df['image_id'].unique()):.2f}")
    st.write(f"이미지당 최대 어노테이션 수: {annotation_counts.max()}")
    st.write(f"이미지당 최소 어노테이션 수: {annotation_counts.min()}")

    st.write("\n")
    # 2. 이미지당 고유 클래스 수 분포
    st.subheader("2. 이미지당 고유 클래스 수 분포")
    st.text(": 한 이미지의 몇 개의 unique한 class가 있는지, 그 분포를 나타낸 결과")
    
    unique_classes_per_image = df.groupby('image_id')['category_id'].nunique()
    fig2 = px.histogram(unique_classes_per_image, 
                        log_y=True, color_discrete_sequence=['skyblue'], opacity=0.7,
                        labels={"value":"Number of Unique Classes"},
                        title="<b>DISTRIBUTION OF # OF UNIQUE CLASSES PER IMAGE   " +
                              "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
                        )
    fig2.update_layout(showlegend=False,
                       xaxis_title="<b>Number of Unique Classes</b>",
                       yaxis_title="<b>Count of Images</b>",)
    
    st.plotly_chart(fig2, use_container_width=True)
    st.write(f"총 고유 클래스 수: {len(df['category_id'].unique())}")
    st.write(f"이미지당 평균 고유 클래스 수: {unique_classes_per_image.mean():.2f}")
    st.write(f"이미지당 최대 고유 클래스 수: {unique_classes_per_image.max()}")
    st.write(f"이미지당 최소 고유 클래스 수: {unique_classes_per_image.min()}")

    st.write("\n")
    # 3. 클래스별 어노테이션 수 분포
    st.subheader("3. 클래스별 어노테이션 수 분포")

    st.text(": 각 Class 당 몇 개의 Annotation, 즉 bbox가 있는지, 그 분포를 나타낸 결과")
    
    class_counts = df['category_id'].value_counts()
    category_names = {cat['id']: cat['name'] for cat in train_annotations['categories']}
    class_counts.index = class_counts.index.map(category_names)
    
    fig3 = px.bar(x=class_counts.index, y=class_counts.values,
                  labels={'x': 'Class Name', 'y': 'Number of Annotations'},
                  title="<b>DISTRIBUTION OF ANNOTATIONS PER CLASS</b>",
                  color=class_counts.values, color_continuous_scale='Viridis')
    fig3.update_layout(xaxis_tickangle=-45, showlegend=False)
    
    st.plotly_chart(fig3, use_container_width=True)

    st.write("\n")
    # **추가 부분: Bounding Box 면적 분포**
    # 이미지 크기 정보를 bbox_df에 추가
    bbox_df = df.copy()  # 어노테이션 데이터프레임 복사
    image_sizes = {img['id']: (img['width'], img['height']) for img in train_annotations['images']}
    bbox_df['image_size'] = bbox_df['image_id'].map(image_sizes)

    # BBox 영역 비율 계산
    bbox_df["frac_bbox_area"] = bbox_df.apply(lambda row: (row["bbox"][2] * row["bbox"][3]) / (row['image_size'][0] * row['image_size'][1]), axis=1)
    bbox_df["class_name"] = bbox_df["category_id"].map(category_names)

    # Bounding Box Area 분포 그래프
    st.subheader("4. Distribution of Bounding Box Areas")
    st.text(": 각 Class 별 bounding box의 크기 분포")

    fig4 = px.box(bbox_df.sort_values(by="class_name"), x="class_name", y="frac_bbox_area", color="class_name", 
                 color_discrete_sequence=px.colors.qualitative.Vivid, notched=True,
                 labels={"class_name": "Class Name", "frac_bbox_area": "BBox Area (%)"},
                 title="<b>DISTRIBUTION OF BBOX AREAS AS % OF SOURCE IMAGE AREA</b>")

    fig4.update_layout(showlegend=True, yaxis_range=[-0.025, 0.4],
                       xaxis_title="", yaxis_title="<b>Bounding Box Area %</b>")
    st.plotly_chart(fig4, use_container_width=True)


    st.subheader("5. Aspect Ratio for bounding boxes by class")
    st.text(": 각 Class 별 bounding box의 aspect ratio 값입니다. 이를 활용하여 anchor generator의 aspect ratio 비율을 조절할 수 있습니다.")

    # 'x_max'와 'y_max' 계산
    bbox_df["x_max"] = bbox_df["bbox"].apply(lambda x: x[0] + x[2])  # x_min + width
    bbox_df["y_max"] = bbox_df["bbox"].apply(lambda x: x[1] + x[3])  # y_min + height
    bbox_df["x_min"] = bbox_df["bbox"].apply(lambda x: x[0])
    bbox_df["y_min"] = bbox_df["bbox"].apply(lambda x: x[1])

    # Aspect Ratio 계산
    bbox_df["aspect_ratio"] = (bbox_df["x_max"] - bbox_df["x_min"]) / (bbox_df["y_max"] - bbox_df["y_min"])

    # aspect_ratio 열이 float 형태인지 확인하고 변환
    bbox_df["aspect_ratio"] = bbox_df["aspect_ratio"].astype(float)

    # 클래스별 평균 aspect ratio 계산 및 출력
    st.write(bbox_df.groupby("category_id").mean(numeric_only=True)[["aspect_ratio"]])

    # 카테고리 이름 가져오기
    category_names = {cat['id']: cat['name'] for cat in train_annotations['categories']}
    classes = [category_names[cat_id] for cat_id in bbox_df["category_id"].unique()]

    # Aspect Ratio에 대한 막대 그래프 생성
    fig = px.bar(x=classes, y=bbox_df.groupby("category_id").mean(numeric_only=True)["aspect_ratio"], 
                color=classes, opacity=0.85,
                labels={"x":"Class Name", "y":"Aspect Ratio (W/H)"},
                title="<b>Aspect Ratios For Bounding Boxes By Class</b>",)
    fig.update_layout(
                    yaxis_title="<b>Aspect Ratio (W/H)</b>",
                    xaxis_title=None,
                    legend_title_text=None)
    fig.add_hline(y=1, line_width=2, line_dash="dot", 
                annotation_font_size=10, 
                annotation_text="<b>SQUARE ASPECT RATIO</b>", 
                annotation_position="bottom left", 
                annotation_font_color="black")
    fig.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="red", opacity=0.125,
                annotation_text="<b>>2:1 VERTICAL RECTANGLE REGION</b>", 
                annotation_position="bottom right", 
                annotation_font_size=10,
                annotation_font_color="red")
    fig.add_hrect(y0=2, y1=3.5, line_width=0, fillcolor="green", opacity=0.04,
                annotation_text="<b>>2:1 HORIZONTAL RECTANGLE REGION</b>", 
                annotation_position="top right", 
                annotation_font_size=10,
                annotation_font_color="green")

    st.plotly_chart(fig, use_container_width=True)



# 사이드바에 추가 정보 표시
st.sidebar.info("Object Detection 데이터셋을 분석하기 위한 도구")
st.sidebar.header("전체 데이터셋의 대한 정보")
st.sidebar.text("전체 이미지 : 9754장")
st.sidebar.text("train 이미지 : 4883장")
st.sidebar.text("test 이미지 : 4871장")
st.sidebar.text("클래스 수 : 10개")
st.sidebar.text("어노테이션 수 : 23144개")
st.sidebar.text("폴더 경로 : /data/ephemeral/home/data/dataset")
st.sidebar.text("classe명\n General trash\n Paper\n Paper pack\n Metal\n Glass\n Plastic\n Styrofoam\n Plastic bag\n Battery\n Clothing\n")

# streamlit run /data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/EDA/Jeongseon/eda.py