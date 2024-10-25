
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer
import sys
import os
from pycocotools.coco import COCO
import TTA
from eval import test_eval

py_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(py_dir_path, "../model_train"))
import utils
import dataset


def main(model_rel_path):
    model_path = os.path.join(py_dir_path, model_rel_path)
    coco_dir_path = os.path.join(py_dir_path, '../../../Split_data')
    data_dir_path = os.path.join(py_dir_path, '../../../../data/dataset')
    test_info_name = 'test.json'
    device = 'cuda'

    checkpoint_path = utils.find_checkpoint_path(model_path)
    run_name = os.path.split(model_path)[-1]
    json_path = os.path.join(model_path, run_name + '.json')

    # conf 파일 경로 덮어쓰기
    conf = utils.read_json(json_path)
    conf['coco_dir_path'] = coco_dir_path
    conf['data_dir_path'] = data_dir_path


    valid_info_path = os.path.join(conf['coco_dir_path'], conf['valid_info_name'])
    coco_valid = COCO(valid_info_path)
    valid = dataset.COCO2dataset(conf['data_dir_path'], coco_valid)

    test_info_path = os.path.join(conf['data_dir_path'], test_info_name)
    coco_test = COCO(test_info_path)
    test = dataset.COCO2dataset(conf['data_dir_path'], coco_test)

    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint_path)
    model.to(device)

    tta_transforms_list = [
            (TTA.Compose([TTA.horizontal_flip]), 'hflip'),
            (TTA.Compose([TTA.identity]), 'identity')
        ]

    select = 'test'
    save_dir_path = os.path.join(py_dir_path, f'result/TTA/{select}')
    os.makedirs(save_dir_path, exist_ok=True)

    if select == 'test':
        for tta_transforms, tta_info in tta_transforms_list:
            save_path = f'{save_dir_path}/{run_name}_{tta_info}.csv'
            save_path = utils.renew_if_path_exist(save_path)
            test_eval(model, image_processor, coco_test, test, tta_transforms, save_path)


if __name__ == '__main__':
    # TTA할 모델 경로 지정
    # 저장된 모델을 불러와서 tta_transforms을 수행하고 각 결과를 저장
    # 예시: 
    # tta_transforms_list = [
    #        (TTA.Compose([TTA.horizontal_flip]), 'hflip'),
    #        (TTA.Compose([TTA.identity]), 'identity')
    #    ]
    # 출력 파일:
    # deta-swin-large_0_img_size_720_identity.csv, deta-swin-large_0_img_size_720_hflip.csv

    fold_path = '../model_train/result/jozhang97'  
    model_paths = ['deta-swin-large_0_img_size_720', 'deta-swin-large_1_img_size_720', 'deta-swin-large_2_img_size_680',
                  'deta-swin-large_3_img_size_720', 'deta-swin-large_4_img_size_720']  
    
    for model_path in model_paths:
        path = os.path.join(fold_path, model_path)
        main(path)