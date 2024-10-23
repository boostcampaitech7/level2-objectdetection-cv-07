# visualize_results.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate model using COCOeval and plot Precision-Recall curve')
    parser.add_argument('pkl_file', help='Path to the result pkl file')
    parser.add_argument('ann_file', help='Path to the COCO annotation file')
    parser.add_argument(
        '--work-dir',
        help='The directory to save the evaluation metrics and PR curve',
        default='./work_dir')
    return parser.parse_args()

def plot_precision_recall_curve(coco_eval, class_ids, class_names, iou_thr=0.5, mAP=None):
    """각 클래스에 대한 Precision-Recall Curve 그리기 함수"""
    precision = coco_eval.eval['precision']
    iou_thr_idx = np.where(np.isclose(coco_eval.params.iouThrs, iou_thr))[0][0]  # 특정 IoU에 대한 인덱스 가져오기
    
    plt.figure(figsize=(10, 8))
    
    for idx, class_id in enumerate(class_ids):
        pr_curve = precision[iou_thr_idx, :, class_id, 0, 2]  # 특정 클래스와 IoU에서의 PR Curve
        recall = np.linspace(0, 1, pr_curve.shape[0])

        # AP 계산
        ap = np.mean(pr_curve)
        
        # 클래스 이름과 AP를 레이블에 포함
        class_name = class_names[idx]
        plt.plot(recall, pr_curve, label=f'{class_name} (AP={ap:.2f}, IoU={iou_thr})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # mAP를 제목에 포함
    plt.title(f'Precision-Recall Curve for Classes (IoU={iou_thr})\nmAP: {mAP:.2f}')
    
    # legend를 그래프 밖에 배치
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(False)

    # 이미지 저장
    output_image_path = '/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/work_dirs/co_dino/plot/precision_recall_curve.png'
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(output_image_path, bbox_inches='tight')  # 그래프를 저장할 때 legend 포함하여 저장
    plt.show()

def convert_to_coco_format(results, img_ids):
    """pkl 파일의 예측 결과를 COCO 형식으로 변환"""
    coco_results = []
    for i, pred in enumerate(results):
        pred_instances = pred['pred_instances']
        for j in range(len(pred_instances['scores'])):
            coco_result = {
                'image_id': img_ids[i],  # 해당 이미지의 ID
                'category_id': int(pred_instances['labels'][j]),  # 클래스 ID
                'bbox': pred_instances['bboxes'][j].tolist(),  # 바운딩 박스
                'score': float(pred_instances['scores'][j])  # 예측 점수
            }
            # 바운딩 박스를 COCO 형식으로 변환 (x, y, w, h)
            coco_result['bbox'][2] -= coco_result['bbox'][0]  # width
            coco_result['bbox'][3] -= coco_result['bbox'][1]  # height
            coco_results.append(coco_result)
    return coco_results

def main():
    args = parse_args()

    # COCO dataset 불러오기
    coco = COCO(args.ann_file)

    # pkl 파일 로드 (예측 결과)
    with open(args.pkl_file, 'rb') as f:
        results = pickle.load(f)

    # 이미지 ID 불러오기
    img_ids = coco.getImgIds()

    # 예측 결과를 COCO 형식으로 변환
    coco_results = convert_to_coco_format(results, img_ids)

    # COCO 형식의 예측 결과를 로드
    coco_dt = coco.loadRes(coco_results)

    # COCOeval 객체 생성 및 평가 수행
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 전체 mAP 계산 (평균 AP)
    mAP = coco_eval.stats[0]  # COCOeval의 첫 번째 요소가 mAP (IoU=0.5:0.95)

    # 클래스 메타 정보 (클래스 이름)
    metainfo = {'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')}

    # COCO에 있는 클래스 ID 가져오기
    class_ids = coco.getCatIds()

    # 클래스별 Precision-Recall Curve 그리기, mAP 추가
    plot_precision_recall_curve(coco_eval, class_ids, metainfo['classes'], iou_thr=0.5, mAP=mAP)

if __name__ == '__main__':
    main()
