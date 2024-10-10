from mmcv import Config
from mmdet.utils import get_device

def load_config(config_file):
    return Config.fromfile(config_file)

def update_config(cfg, root, classes):
    # Update train dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = f'{root}/train_split.json'
    cfg.data.train.pipeline[4]['img_scale'] = (512, 512)

    # Update validation dataset config
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = f'{root}/val_split.json'
    cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    # Update test dataset config
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = f'{root}/test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)
    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = 'mmdetection/work_dirs/yolov3_trash'

    cfg.model.bbox_head.num_classes = 10
    cfg.runner.max_epochs = 15

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    return cfg