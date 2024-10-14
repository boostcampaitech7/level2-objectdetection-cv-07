from mmcv import Config
import json
from mmdet.utils import get_device

def load_config(config_file):
    return Config.fromfile(config_file)

def update_config(cfg, dataset_root, split_dataset_root, work_dir, classes, k_folds, curr_fold):
    # Update train dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = dataset_root
    cfg.data.train.ann_file = f'{split_dataset_root}/train_{curr_fold}_{k_folds}.json'
    cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

    # Update validation dataset config
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = dataset_root
    cfg.data.val.ann_file = f'{split_dataset_root}/val_{curr_fold}_{k_folds}.json'
    cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    # Update test dataset config
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = dataset_root
    cfg.data.test.ann_file = f'{dataset_root}/test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)
    cfg.data.samples_per_gpu = 4    

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = f'{work_dir}/fold_{curr_fold}'

    cfg.model.roi_head.bbox_head.num_classes = len(classes)
    cfg.runner.max_epochs = 15

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    
    # 수정된 config JSON 파일 저장
    cfg_dict = cfg._cfg_dict.to_dict()
    with open(f'mmdetection/model_configs/config_fold_{curr_fold}.json', 'w') as f:
        json.dump(cfg_dict, f, indent=4)
        
    return cfg