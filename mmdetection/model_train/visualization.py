import numpy as np
import matplotlib.pyplot as plt
import mmcv
from mmdet.evaluation import COCOEval
from sklearn.metrics import precision_recall_curve, average_precision_score
# 1. 저장된 예측 결과(pkl)와 annotation 파일 불러오기
results = mmcv.load('results.pkl')
ann_file = 'path/to/annotation_file.json'  # COCO 형식의 annotation 파일 경로
# 2. COCO 평가자 생성 및 결과 처리
coco_eval = COCOEval(ann_file, iou_type='bbox')
# 결과를 COCO 형식으로 변환하여 평가 수행
coco_eval.results2json(results, outfile_prefix='results')
coco_eval.evaluate()
# 3. COCO API로 Precision-Recall Curve 계산
precision = coco_eval.eval['precision']
# precision shape: (iou_thresholds, recall, classes, area_range, max_dets)
# 특정 클래스 및 IoU 값에 대한 Precision-Recall Curve 추출
iou_thr = 0.5  # IoU threshold 설정 (0.5로 고정)
cls_idx = 0  # 첫 번째 클래스에 대한 PR Curve (0번째 클래스에 대해 그리기)
# 특정 IoU에 대한 precision-recall 추출
pr_curve = precision[iou_thr, :, cls_idx, 0, 2]  # IoU 0.5에서의 Precision-Recall Curve
# Recall 값 생성
recall = np.linspace(0, 1, pr_curve.shape[0])
# 4. Precision-Recall Curve 그리기
plt.plot(recall, pr_curve, label=f'Precision-Recall Curve (IoU={iou_thr}, Class={cls_idx})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()