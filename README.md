<div align='center'>
  <h2>🏆 재활용 품목 분류를 위한 Object Detection</h1>
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

## Project Overview
#### **Timeline (9/30 - 10/24)**
1. EDA 및 baseline code 분석
2. Baseline model 실험
3. 앙상블
4. 최종 결과 분석

#### **최종 결과(추후 수정)**
<img width="80%" alt="최종 리더보드 순위" src="https://github.com/user-attachments/assets/e5e90019-dda0-4753-9df1-b70ad4174f9b">

## Final Models

Model | Backbone | Lr schd | 더 넣고 싶은 특징!! | box mAP50 |                        Config                         |                                                                                                                                                                    Download                                                                                                                                                                    |
| :------: | :---------: | :-----: | :----------: | :----: | :---------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Co-Dino   | R-50  |   12e   |         |    |     [config](./dino-4scale_r50_8xb2-12e_coco.py)      |                   [model](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705.log.json)                   |
|  Deta  | Swin-L |   36e   |         |    |    [config](./dino-5scale_swin-l_8xb2-36e_coco.py)    |                                                 [model](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth) \| [log](https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/20230307_032359.log)                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |
|    |  |      |         |    |    [config]()    |                                                 [model]() \| [log]()                                                 |

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
Step 1. We need to download config and checkpoint files.
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

Step 2. Verify the inference demo.
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

```

## File Tree
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

## Environment
OS : Linux-5.4.0 <br>
GPU : Tesla V100 (32GB) <br>
Python Version: 3.10.13 <br>
IDE: Visual Studio Code <br>
Tool : Github, Slack, Notion, Zoom <br>
Experiment Tracking: Weights and Biases (WandB)

## Team Members
<div align='center'>
  <h3>럭키비키🍀</h3>
  <table width="98%">
    <tr>
      <td align="center" valign="top" width="15%"><a href="https://github.com/jinlee24"><img src="https://avatars.githubusercontent.com/u/137850412?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/stop0729"><img src="https://avatars.githubusercontent.com/u/78136790?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/yjs616"><img src="https://avatars.githubusercontent.com/u/107312651?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/sng-tory"><img src="https://avatars.githubusercontent.com/u/176906855?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/Soojeoong"><img src="https://avatars.githubusercontent.com/u/100748928?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/cyndii20"><img src="https://avatars.githubusercontent.com/u/90389093?v=4"></a></td>
    </tr>
    <tr>
      <td align="center">이동진</td>
      <td align="center">정지환</td>
      <td align="center">유정선</td>
      <td align="center">신승철</td>
      <td align="center">김소정</td>
      <td align="center">서정연</td>
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
