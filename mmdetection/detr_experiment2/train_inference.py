import sys
import os


# 상위 폴더를 sys.path에 추가
sys.path.append('/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/Co-DETR-main')




from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO
from projects import *
import pandas as pd
import importlib.util




def list_files_in_folder(folder_path):
    # 폴더 내 파일 이름을 리스트로 반환
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # 파일만 추가 (디렉토리는 제외)
            files.append(file_name)
    return files

# 사용 예시
folder_path = '/data/ephemeral/home/Jihwan/level2-objectdetection-cv-07/Co-DETR-main/detr_experiment2/configs'  # 파일이 있는 폴더 경로, 바꿔 줘야 할 부분 ★
files = list_files_in_folder(folder_path)



for py_file in files:
    file_path = os.path.join(folder_path, py_file)
    
    # 모듈 이름을 파일 이름에서 확장자(.py)를 제외하고 설정
    module_name = py_file[:-3]  # .py 확장자 제거
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # 모듈 실행
    
    func = getattr(module, 'get_config')




    ########## train 부분
    cfg = func()
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets[0], cfg, distributed=False, validate=False)




    ########## test 부분
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True


    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    epoch = 'latest'

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

    import pandas as pd

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds() 

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'{module_name}_{epoch}.csv'), index=None)
