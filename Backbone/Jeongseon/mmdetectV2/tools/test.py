import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO

def setup_test_config(config_file, root):
    cfg = Config.fromfile(config_file)
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    cfg.data.test.test_mode = True

    return cfg

def build_and_evaluate_model(cfg, checkpoint_path):
    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)
    return output, dataset

def save_results_to_csv(output, dataset, work_dir, epoch):
    prediction_strings = []
    file_names = []
    coco = COCO(dataset.ann_file)
    
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(10):  # class_num = 10
            for o in out[j]:
                prediction_string += f"{j} {o[4]} {o[0]} {o[1]} {o[2]} {o[3]} "
                
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame({'PredictionString': prediction_strings, 'image_id': file_names})
    submission.to_csv(os.path.join(work_dir, f'submission_{epoch}.csv'), index=False)

def main():
    root = '/data/ephemeral/home/data/dataset/'
    cfg = setup_test_config('./configs/swin/faster_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py', root)

    epoch = 'latest'
    cfg.work_dir = './work_dirs/ssd300_coco_trash'    #
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    output, dataset = build_and_evaluate_model(cfg, checkpoint_path)
    save_results_to_csv(output, dataset, cfg.work_dir, epoch)

if __name__ == '__main__':
    main()
