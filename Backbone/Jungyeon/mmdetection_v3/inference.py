import json
import os
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

# JSON 파일에서 설정값 불러오기
with open('work_dirs/coco_detection/test.bbox.json', 'r') as f:
    output = json.load(f)
    
def inference(output):
    prediction_strings = []
    file_names = []
    coco = COCO('/data/ephemeral/home/dataset/test.json')
    img_ids = coco.getImgIds()

    for id in range(len(img_ids)):
        # class_num = 10
        image_info = coco.loadImgs(coco.getImgIds(imgIds=id))[0]
        prediction_string = ''
        for i, out in enumerate(output):
            if int(out['image_id']) == id:
                prediction_string += str(out['category_id']) + ' ' + str(out['score']) + ' ' + str(out['bbox'][0]) + ' ' + str(out['bbox'][1]) + ' ' + str(out['bbox'][2]) + ' ' + str(out['bbox'][3]) + ' '
                # for j in range(class_num):
                #     for o in out[j]:
                #         print(o)
                #         break
                        # prediction_string += str(out['category_id']) + ' ' + str(out['score']) + ' ' + str(out['bbox'][0]) + ' ' + str(out['bbox'][1]) + ' ' + str(
                        #     out['bbox'][2]) + ' ' + str(out['bbox'][3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    print(len(prediction_strings))
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join('submisson', f'submission_{epoch}.csv'), index=None)
    submission.head()
    return submission

if __name__ == '__main__':
    epoch = 'latest'
    # Inference 실행
    inference(output)