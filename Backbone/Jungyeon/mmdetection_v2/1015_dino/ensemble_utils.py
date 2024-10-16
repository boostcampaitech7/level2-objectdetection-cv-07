import pandas as pd
import numpy as np
from ensemble_boxes import nms
from pycocotools.coco import COCO


def load_submissions(submission_files):
    """CSV 파일들을 읽어와서 DataFrame으로 반환"""
    return [pd.read_csv(file) for file in submission_files]


def prepare_boxes(submission_df, image_id, coco, i, image_info):
    """하나의 이미지에 대한 박스, 스코어, 라벨 리스트 준비"""
    boxes_list = []
    scores_list = []
    labels_list = []

    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()

        if len(predict_list) == 0 or len(predict_list) == 1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        for box in predict_list[:, 2:6].tolist():
            # 박스 좌표 정규화
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))

    return boxes_list, scores_list, labels_list


def ensemble_boxes_nms(boxes_list, scores_list, labels_list, iou_thr, image_info):
    """NMS를 적용하여 박스, 스코어, 라벨 결과 반환"""
    if len(boxes_list):
        boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
        return boxes, scores, labels
    return [], [], []


def generate_submission_file(prediction_strings, image_ids, output_file):
    """제출용 파일 생성"""
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = image_ids
    submission.to_csv(output_file, index=None)
    return submission