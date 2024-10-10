from inference_utils import run_inference

if __name__ == '__main__':
    config_path = 'mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py'
    root = '/data/ephemeral/home/dataset'
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    epoch = 'latest'

    # Inference 실행
    run_inference(config_path, root, classes, epoch)