import pandas as pd
import numpy as np
import ensemble_boxes 
from pycocotools.coco import COCO
import itertools
import os
from tqdm import tqdm

def get_box_score_label(submission_df, image_id, image_info):
    boxes_list = []
    scores_list = []
    labels_list = []

    # 각 submission file 별로 prediction box좌표 불러오기
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()

        if len(predict_string) == 0:
            continue

        predict_string = predict_string[0]
        predict_list = str(predict_string).split()

        if len(predict_list) == 0 or len(predict_list) == 1:
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

def ensemble(conf):
    submissions_df = []
    for transform in conf['transforms']:
        submission_file_path = os.path.join(conf['submission_fold_path'], 
                                            conf['submission_file_format'].format(mode=conf['mode'], transform=transform))
        submission_df = pd.read_csv(submission_file_path)
        submissions_df.append(submission_df)

    image_ids = submissions_df[0]['image_id'].tolist()
    coco = COCO(conf['annotation_path'])

    exp = list(enumerate(list(itertools.product(conf['algorithms'], conf['iou_thresholds'])))) # ensemble할 조건 리스트
    results = [{'prediction_strings': [], 'file_names': []} for i in range(len(exp))] # 결과 저장 리스트 선언

    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(tqdm(image_ids)):    
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []

        image_info = coco.loadImgs(int(image_id[6:-4]))[0]
        image_width = image_info['width']
        image_height = image_info['height']

        boxes_list, scores_list, labels_list = get_box_score_label(submissions_df, image_id, image_info)
        
        # 예측 box가 있다면 ensemble 수행
        if len(boxes_list):
            # ensemble에 필요한 인자: [box의 lists, confidence score의 lists, label의 list, iou에 사용할 threshold]
            for j, (algorithm, iou_threshold) in exp:
                boxes, scores, labels = getattr(ensemble_boxes, algorithm)(boxes_list, scores_list, labels_list, iou_thr=iou_threshold)
                prediction_string = get_prediction(boxes, scores, labels, image_width, image_height)

                results[j]['prediction_strings'].append(prediction_string)
                results[j]['file_names'].append(image_id)    

    # 결과 저장하기
    os.makedirs(conf['output_fold_path'], exist_ok=True)

    for i, (algorithm, iou_threshold) in exp:
        output_path = conf['output_path_format'].format(algorithm=algorithm, iou_threshold=iou_threshold)

        submission = pd.DataFrame()
        submission['PredictionString'] = results[i]['prediction_strings']
        submission['image_id'] = results[i]['file_names']

        submission.to_csv(output_path, index=False)
        submission.head()

def ensemble_temp(conf):
    submissions_df = []
    for submission_file_path in conf['submission_file_paths']:
        submission_df = pd.read_csv(submission_file_path)
        submissions_df.append(submission_df)

    image_ids = submissions_df[0]['image_id'].tolist()
    coco = COCO(conf['annotation_path'])

    exp = list(enumerate(list(itertools.product(conf['algorithms'], conf['iou_thresholds'])))) # ensemble할 조건 리스트
    results = [{'prediction_strings': [], 'file_names': []} for i in range(len(exp))] # 결과 저장 리스트 선언

    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(tqdm(image_ids)):    
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []

        image_info = coco.loadImgs(int(image_id[6:-4]))[0]
        image_width = image_info['width']
        image_height = image_info['height']

        boxes_list, scores_list, labels_list = get_box_score_label(submissions_df, image_id, image_info)
        
        # 예측 box가 있다면 ensemble 수행
        if len(boxes_list):
            # ensemble에 필요한 인자: [box의 lists, confidence score의 lists, label의 list, iou에 사용할 threshold]
            for j, (algorithm, iou_threshold) in exp:
                boxes, scores, labels = getattr(ensemble_boxes, algorithm)(boxes_list, scores_list, labels_list, iou_thr=iou_threshold)
                prediction_string = get_prediction(boxes, scores, labels, image_width, image_height)

                results[j]['prediction_strings'].append(prediction_string)
                results[j]['file_names'].append(image_id)    

    # 결과 저장하기
    os.makedirs(conf['output_fold_path'], exist_ok=True)

    for i, (algorithm, iou_threshold) in exp:
        output_path = conf['output_path_format'].format(algorithm=algorithm, iou_threshold=iou_threshold)

        submission = pd.DataFrame()
        submission['PredictionString'] = results[i]['prediction_strings']
        submission['image_id'] = results[i]['file_names']

        submission.to_csv(output_path, index=False)
        submission.head()
