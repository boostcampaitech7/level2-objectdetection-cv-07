<div align='center'>
  <img width="524" alt="Screenshot 2024-10-21 at 10 08 24 PM" src="https://github.com/user-attachments/assets/d7db3331-a5fe-49d6-b107-82cc10bd42d1">  
  <h2>🏆 재활용 품목 분류를 위한 Object Detection</h2>
</div>

<div align="center">


[📘Wrap-Up Report](https://detrex.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) |
[👀Model](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) |
[🤔Issues](https://github.com/boostcampaitech7/level2-objectdetection-cv-07/issues) | <br>
[🚀MMDetection](https://github.com/open-mmlab/mmdetection) |
[🤗Transformers](https://huggingface.co/docs/transformers/en/index) |

</div>

## Introduction
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>

**Goal :** 쓰레기 객체를 탐지하는 모델을 개발하여 정확한 분리수거와 환경 보호를 지원 <br>
**Data :** COCO format의 쓰레기 객체 이미지 9754장

<br>

## Project Overview
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>
**Timeline ⏳ (9/30 - 10/24)**<br>
EDA 및 baseline code 분석 → Baseline model 실험 → 앙상블 → 최종 결과 분석<br>&emsp;

**최종 결과 📈(추후 수정)**
<br>
<img align="center" width="80%" alt="최종 리더보드 순위" src="https://github.com/user-attachments/assets/e5e90019-dda0-4753-9df1-b70ad4174f9b">

## Final Model
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>
Model | Backbone | Lr schd | 더 넣고 싶은 특징!! | box mAP50 |                        Config                         |  Download  |
| :------: | :---------: | :-----: | :----------: | :----: | :---------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Co-Dino   | R-50  |   12e   |         |    |     [config](./dino-4scale_r50_8xb2-12e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705.log.json)                   |
|  Deta  | Swin-L |   36e   |         |    |    [config](./dino-5scale_swin-l_8xb2-36e_coco.py)    |                                                 [model](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth) \| [log](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/20230307_032359.log)                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |

## Installation Guide
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>
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
Step 1. We need to download config and checkpoint files.
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

Step 2. Verify the inference demo.
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

```
## File Tree(추후 수정)
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>
```
  ├─.github
  ├─ mmdetection
    ├─config 파일
    ├─checkpoint 파일
    ├─test 파일
    ├─train 파일
  ├─tranformers
    ├─config 파일
    ├─checkpoint 파일
    ├─test 파일
    ├─train 파일
  ├─ensemble_inference.py
  ├─demo
    ├─model_demo.py
  ├─requirements.txt
  ├─README.md
```
## Environment Setting(추후수정)
많은 물건이 대량으로 생산되고, 소비되는 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되기에 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. <br>
**1. System Setup & Libraries**
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
    <td>Docker</td>
    <td>Linux 5.4.0</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.8.13</td>
    <td>Git</td>
    <td>3.8.13</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>NVIDIA RTX 3090</td>
    <td>Conda</td>
    <td>NVIDIA RTX 3090</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>NVIDIA RTX 3090</td>
    <td>Tmux</td>
    <td>NVIDIA RTX 3090</td>
  </tr>
  <tr>
    <td>CUDNN</td>
    <td>NVIDIA RTX 3090</td>
    <td>OS</td>
    <td>NVIDIA RTX 3090</td>
  </tr>
</table>
<br>

**2. Dependencies**  
아래 주요 라이브러리 및 버전이 필요합니다. 전체 목록은 [requirements.txt](./requirements.txt)에서 확인할 수 있습니다.
- `torch==1.12.1`  
- `torchvision==0.13.1`  
- `numpy==1.21.2`  
- `pandas==1.3.3`  
- `scikit-learn==0.24.2`

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
