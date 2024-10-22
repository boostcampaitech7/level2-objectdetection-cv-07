from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

from .mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

## custom 설정
dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False
dataloader.train.dataset.names='coco_trash_train'
dataloader.test.dataset.names='coco_trash_test'
dataloader.evaluator.dataset_name='coco_trash_test'


# del model.roi_heads['mask_in_features']
# del model.roi_heads['mask_pooler']
# del model.roi_heads['mask_head']
model.roi_heads.num_classes=10

train.eval_period=500
train.output_dir='/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/vitdet_rcnn_cascade/output'
train.checkpointer.max_to_keep=5
train.checkpointer.period=5000

# ## 증강
# dataloader.train.mapper.augmentations = [
#     #{"_target_": "detectron2.data.transforms.ResizeShortestEdge", "max_size": 768, "short_edge_length": 768},
#     {"_target_": "detectron2.data.transforms.RandomFlip", "prob" : 0.5, "horizontal" : False ,"vertical" : True},
#     {"_target_": "detectron2.data.transforms.RandomBrightness", "intensity_min" : 0.8, "intensity_max" : 1.8},
#     {"_target_": "detectron2.data.transforms.RandomContrast", "intensity_min" : 0.6, "intensity_max" : 1.3}
# ]

# dataloader.test.mapper.augmentations = [
#     #{"_target_": "detectron2.data.transforms.ResizeShortestEdge", "max_size": 768, "short_edge_length": 768}
# ]
# dataloader.train.num_workers=4
# dataloader.test.num_workers=4
### batch size
dataloader.train.total_batch_size=2
# model.proposal_generator.batch_size_per_image=128
# model.roi_heads.batch_size_per_image=128