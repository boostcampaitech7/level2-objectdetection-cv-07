from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)

dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False
dataloader.train.dataset.names='coco_trash_train'
dataloader.test.dataset.names='coco_trash_test'
dataloader.evaluator.dataset_name='coco_trash_test'

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

del model.roi_heads['mask_in_features']
del model.roi_heads['mask_pooler']
del model.roi_heads['mask_head']
model.roi_heads.num_classes=10

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

train.eval_period=5000
train.output_dir='/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/vitdet_rcnn/output'

## 증강
dataloader.train.mapper.augmentations = [
    {"_target_": "detectron2.data.transforms.RandomFlip", "prob" : 0.5, "horizontal" : False ,"vertical" : True},
    {"_target_": "detectron2.data.transforms.RandomBrightness", "intensity_min" : 0.8, "intensity_max" : 1.8},
    {"_target_": "detectron2.data.transforms.RandomContrast", "intensity_min" : 0.6, "intensity_max" : 1.3}
]

dataloader.test.mapper.augmentations = []

### batch size
dataloader.train.total_batch_size=4
model.proposal_generator.batch_size_per_image=128
model.roi_heads.batch_size_per_image=128
