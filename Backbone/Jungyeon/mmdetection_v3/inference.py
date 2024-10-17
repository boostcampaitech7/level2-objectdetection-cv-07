import json
import os
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

# JSON 파일에서 설정값 불러오기
with open('work_dirs/coco_detection/test.bbox.json', 'r') as f:
    output = json.load(f)
    
def coco_to_voc(coco_bbox):
    x_min = coco_bbox[0]
    y_min = coco_bbox[1]
    x_max = coco_bbox[0] + coco_bbox[2]  # x_min + width
    y_max = coco_bbox[1] + coco_bbox[3]  # y_min + height
    return [x_min, y_min, x_max, y_max]
    
def inference(output):
    prediction_strings = []
    file_names = []
    coco = COCO('/data/ephemeral/home/dataset/test.json')
    img_ids = coco.getImgIds()

    for id in range(len(img_ids)):
        print(f'=======================current id:{id}=======================')
        image_info = coco.loadImgs(coco.getImgIds(imgIds=id))[0]
        prediction_string = ''
        last_idx = 0
        for i, out in enumerate(output):
            if int(out['image_id']) == id:
                last_idx = i
                pas_voc_bbox = coco_to_voc(out['bbox'])
                prediction_string += str(out['category_id']) + ' ' + str(out['score']) + ' ' + str(pas_voc_bbox[0]) + ' ' + str(pas_voc_bbox[1]) + ' ' + str(pas_voc_bbox[2]) + ' ' + str(pas_voc_bbox[3]) + ' '
        print(f'last index:{last_idx}')
        print(f'finished:{id}')
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    print(len(prediction_strings))
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join('./', 'submission_pas.csv'), index=None)
    submission.head()
    return

if __name__ == '__main__':
    epoch = 'latest'
    # Inference 실행
    inference(output)