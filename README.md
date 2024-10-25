<div align='center'>
  <img width="524" alt="Screenshot 2024-10-21 at 10 08 24 PM" src="https://github.com/user-attachments/assets/d7db3331-a5fe-49d6-b107-82cc10bd42d1">  
  <h2>🏆 재활용 품목 분류를 위한 Object Detection</h2>
</div>

<div align="center">


[📘Wrap-Up Report]() |
[👀Model](#final-model) |
[🛠️Installation](#installation-guide) |
[🤔Issues](https://github.com/boostcampaitech7/level2-objectdetection-cv-07/issues) | <br>
[🚀MMDetection](https://github.com/open-mmlab/mmdetection) |
[🤗Transformers](https://huggingface.co/docs/transformers/en/index) |
[💎Detectron2](https://github.com/facebookresearch/detectron2) |
</div>

## Introduction
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>

**Goal :** 쓰레기 객체를 탐지하는 모델을 개발하여 정확한 분리수거와 환경 보호를 지원 <br>
**Data :** COCO format의 쓰레기 객체 이미지 9754장<br>
**Metric :** Test set의 mAP50(Mean Average Precision)

## Project Overview
먼저 EDA와 baseline 모델 분석을 수행한 후, mmdetection, transformers 등의 라이브러리를 활용하여 데이터셋에 대한 다양한 모델의 성능을 실험했습니다. 최종적으로 앙상블 기법을 통해 성능을 극대화하였고, 이를 바탕으로 최종 모델 아키텍처를 구성하여 분석을 진행했습니다.<br> 결과적으로 **mAP50 0.7382**를 달성하여 리더보드에서 3위를 기록하였습니다.<br>

<img width="70%" alt="최종 public 리더보드 순위" src="https://github.com/user-attachments/assets/78a3accd-ed78-4560-bc97-a5c5421089b1"><br>

## Final Model
다음은 최종 모델 구성에 사용된 모델들입니다. 최종적으로 DETA 5-fold 결과와 Co-DINO 5-fold 결과를 기반으로 threshold를 0.7로 설정하여 WBF를 실행한 결과, 최종 성능 **mAP50 0.9999**를 달성했습니다.<br>


|      Model     | Backbone |  Lr schd |   tta  |  k-fold  |  ensemble<br>(threshold)  |   box mAP  |   Configs   |
| :------------: | :------: | :------: | :----: | :------: | :-----------------------: | :--------: | :---------: |
|  Co-Dino       |  R-50    |   12e    |    y   |  4-fold  |          WBF(0.6)         |   0.9999   |  [config]() | 
|  Deta          |  Swin-L  |   36e    |    y   |  5-fold  |          WBF(0.6)         |   0.9999   |  [config]() | 
|  Cascade R-CNN |  MViTv2  |   30e    |    y   |  5-fold  |          WBF(0.7)         |   0.9999   |  [config]() | 

## Data
1. 기본 데이터셋
```
dataset
  ├── annotations
      ├── train.json # train image에 대한 annotation file (coco format)
      └── test.json # test image에 대한 annotation file (coco format)
  ├── train # 4883장의 train image
  └── test # 4871장의 test image
```

## Installation Guide
1. Installation(추후 수정)
```
# Step 1. Create a conda environment and activate it
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Step 2. Install PyTorch following official instructions, e.g.
conda install pytorch torchvision -c pytorch

# Step 3. Install MMEngine and MMCV using MIM.
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Step 4. Install MMDetection.
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
pip install requirements.txt
```
<br>

2. Run Demo(추후 수정)
```
# Step 1. We need to download config and checkpoint files.
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

# Step 2. Verify the inference demo.
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

```
## File Tree(추후 수정)
```
├── .github
├── mmdetection
    ├── configs
    ├── checkpoint
├── tranformers
    ├── configs
    ├── checkpoint
├── demo
    ├── model_demo.py
├── ensemble_inference.py
├── requirements.txt
└── README.md
```
## Environment Setting
<table>
  <tr>
    <th colspan="2">System Information</th> <!-- 행 병합 -->
    <th colspan="2">Tools and Libraries</th> <!-- 열 병합 -->
  </tr>
  <tr>
    <th>Category</th>
    <th>Details</th>
    <th>Category</th>
    <th>Details</th>
  </tr>
  <tr>
    <td>Operating System</td>
    <td>Linux 5.4.0</td>
    <td>Git</td>
    <td>2.25.1</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10.13</td>
    <td>Conda</td>
    <td>23.9.0</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>Tesla V100-SXM2-32GB</td>
    <td>Tmux</td>
    <td>3.0a</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>12.2</td>
    <td></td>
    <td></td>
  </tr>
</table>
<br>

<p align='center'>© 2024 LuckyVicky Team.</p>
<p align='center'>Supported by Naver BoostCamp AI Tech.</p>

---

<div align='center'>
  <h3>👥 Team Members of LuckyVicky</h3>
  <table width="80%">
    <tr>
      <td align="center" valign="top" width="15%"><a href="https://github.com/jinlee24"><img src="https://avatars.githubusercontent.com/u/137850412?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/stop0729"><img src="https://avatars.githubusercontent.com/u/78136790?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/yjs616"><img src="https://avatars.githubusercontent.com/u/107312651?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/sng-tory"><img src="https://avatars.githubusercontent.com/u/176906855?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/Soojeoong"><img src="https://avatars.githubusercontent.com/u/100748928?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/cyndii20"><img src="https://avatars.githubusercontent.com/u/90389093?v=4"></a></td>
    </tr>
    <tr>
      <td align="center">🍀이동진</td>
      <td align="center">🍀정지환</td>
      <td align="center">🍀유정선</td>
      <td align="center">🍀신승철</td>
      <td align="center">🍀김소정</td>
      <td align="center">🍀서정연</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
    </tr>
  </table>
</div>
