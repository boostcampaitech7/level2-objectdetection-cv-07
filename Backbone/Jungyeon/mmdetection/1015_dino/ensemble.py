import json
from ensemble_utils import load_submissions, prepare_boxes, ensemble_boxes_nms, generate_submission_file
from pycocotools.coco import COCO

# JSON 파일에서 설정값 불러오기
with open('mmdetection/ensemble_config.json', 'r') as f:
    config = json.load(f)

def run_ensemble(submission_files, annotation_file, output_file, iou_thr=0.4):
    """Ensemble NMS를 수행하고 결과를 제출 파일로 저장"""
    # Load submissions
    submission_df = load_submissions(submission_files)
    image_ids = submission_df[0]['image_id'].tolist()

    # Load COCO dataset
    coco = COCO(annotation_file)

    prediction_strings = []
    file_names = []

    # Loop through each image
    for i, image_id in enumerate(image_ids):
        print(f"Processing image {i}: {image_id}")
        image_info = coco.loadImgs(i)[0]

        # Prepare boxes, scores, and labels for each image
        boxes_list, scores_list, labels_list = prepare_boxes(submission_df, image_id, coco, i, image_info)

        # Perform NMS
        boxes, scores, labels = ensemble_boxes_nms(boxes_list, scores_list, labels_list, iou_thr, image_info)

        # Generate prediction string
        prediction_string = ''
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += (
                str(label) + ' ' +
                str(score) + ' ' +
                str(box[0] * image_info['width']) + ' ' +
                str(box[1] * image_info['height']) + ' ' +
                str(box[2] * image_info['width']) + ' ' +
                str(box[3] * image_info['height']) + ' '
            )

        # Save prediction results
        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    # Generate submission file
    submission = generate_submission_file(prediction_strings, file_names, output_file)
    print("Ensemble complete. Submission file saved.")

if __name__ == '__main__':
    run_ensemble(config['submission_files'], config['annotation_file'], config['output_file'])