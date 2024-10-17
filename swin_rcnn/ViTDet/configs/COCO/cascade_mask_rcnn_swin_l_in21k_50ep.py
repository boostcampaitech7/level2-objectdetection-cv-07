from .cascade_mask_rcnn_swin_b_in21k_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)


dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False
dataloader.train.total_batch_size=2
dataloader.train.dataset.names='coco_trash_train'
dataloader.test.dataset.names='coco_trash_test'
dataloader.evaluator.dataset_name='coco_trash_test'


model.backbone.bottom_up.depths = [2, 2, 18, 2]
model.backbone.bottom_up.drop_path_rate = 0.4
model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.num_heads = [6, 12, 24, 48]
model.roi_heads.num_classes=10

del model.roi_heads['mask_in_features']
del model.roi_heads['mask_pooler']
del model.roi_heads['mask_head']

train.eval_period=100
train.max_iter=100
train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_large_patch4_window7_224_22k.pth"
train.output_dir='/data/ephemeral/home/Seungcheol/level2-objectdetection-cv-07/swin_rcnn/output'