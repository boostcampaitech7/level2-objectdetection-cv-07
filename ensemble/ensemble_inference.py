import pandas as pd
import numpy as np
import ensemble_boxes 
from pycocotools.coco import COCO
import itertools
import os
from tqdm import tqdm
from copy import deepcopy


def get_box_score_label(submission_df, image_id, image_info):
    boxes_list = []
    scores_list = []
    labels_list = []

    # 각 submission file 별로 prediction box좌표 불러오기
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()
        
        if len(predict_string) == 0:
            if not 'test' in df['image_id'][0]:
                raise Exception(f"Check df['image_id']!: {df['image_id'][0]}")
            continue

        predict_string = predict_string[0]
        predict_list = str(predict_string).split()

        if len(predict_list)==0 or len(predict_list)==1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        for box in predict_list[:, 2:6].tolist():
            # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
            image_width = image_info['width']
            image_height = image_info['height']

            box[0] = float(box[0]) / image_width
            box[1] = float(box[1]) / image_height
            box[2] = float(box[2]) / image_width
            box[3] = float(box[3]) / image_height

            box = np.clip(box, 0, 1).tolist()
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))

    return boxes_list, scores_list, labels_list


def get_prediction(boxes, scores, labels, image_width, image_height):
    prediction_string = ""
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin = box[0] * image_width, box[1] * image_height
        xmax, ymax = box[2] * image_width, box[3] * image_height
        prediction_string += f'{label:.0f} {score:.5f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} '
    return prediction_string


def main(submission_file_paths, annotation_path, output_name, output_fold, algorithms, iou_thresholds, weights):
    submission_df = [pd.read_csv(file) for file in submission_file_paths]
    image_ids = submission_df[0]['image_id'].tolist()
    coco = COCO(annotation_path)

    conf = list(enumerate(list(itertools.product(algorithms, iou_thresholds, weights)))) # ensemble할 조건 리스트
    results = [{'prediction_strings': [], 'file_names': []} for i in range(len(conf))] # 결과 저장 리스트 선언

    # 각 image id 별로 submission file에서 box좌표 추출
    for image_id in tqdm(image_ids):    
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []

        image_id_int = int(image_id[6:-4])
        image_info = coco.loadImgs(image_id_int)[0]
        image_width = image_info['width']
        image_height = image_info['height']

        boxes_list, scores_list, labels_list = get_box_score_label(submission_df, image_id, image_info)
        
        # 예측 box가 있다면 ensemble 수행
        if len(boxes_list):
            # ensemble에 필요한 인자: [box의 lists, confidence score의 lists, label의 list, iou에 사용할 threshold]
            for i, (algorithm, iou_threshold, weight) in conf:
                boxes, scores, labels = getattr(ensemble_boxes, algorithm)(deepcopy(boxes_list), deepcopy(scores_list), deepcopy(labels_list), weights=deepcopy(weight), iou_thr=deepcopy(iou_threshold))
                prediction_string = get_prediction(boxes, scores, labels, image_width, image_height)

                results[i]['prediction_strings'].append(prediction_string)
                results[i]['file_names'].append(image_id)    


    os.makedirs(output_fold, exist_ok=True)
    boxes_list, scores_list, labels_list = get_box_score_label(submission_df, image_id, image_info)

    for i, (algorithm, iou_threshold, weight) in conf:
        weight_str = [f'{i:.3f}' for i in weight]
        new_output_name = output_name + f'_{algorithm}_th_{iou_threshold:.1f}_w_' + '_'.join(weight_str) + '.csv'
        output_path = os.path.join(output_fold, new_output_name)

        submission = pd.DataFrame()
        submission['PredictionString'] = results[i]['prediction_strings']
        submission['image_id'] = results[i]['file_names']

        submission.to_csv(output_path, index=False)
        submission.head()


if __name__ == '__main__':
    # 최종 모델의 앙상블 
    # DETA, co-dino, RCNN의 5-fold 앙상블 결과를 
    # WBF 알고리즘, 모델 가중치 (1.0, 0.4, 0.4), iou 임계값 0.7로 설정하여 최종 앙상블 결과 추출
    
    # 앙상블할 결과 리스트
    submission_file_paths = [
    'ensemble/transformers/TTA/result/ensemble-5fold/deta_ensemble.csv',
    'ensemble/mmdetectionv3/co-dino/co-dino.csv',
    'ensemble/detectron2/rcnn/RCNN.csv',
    ] 

    algorithms = ['weighted_boxes_fusion'] # 알고리즘 설정
    weights = [[1.0, 0.4, 0.4]] # 모델 별 가중치 설정 
    iou_thresholds = [0.7] # iou 임계값
    output_name = f'final_ensemble_threshold_{iou_thresholds[0]}' # 출력파일 이름
    output_fold = 'result/'# 출력 폴더
    json_path = '../data/dataset/test.json' # test dataset json 경로

    main(submission_file_paths, json_path, output_name, output_fold, algorithms, iou_thresholds, weights)


