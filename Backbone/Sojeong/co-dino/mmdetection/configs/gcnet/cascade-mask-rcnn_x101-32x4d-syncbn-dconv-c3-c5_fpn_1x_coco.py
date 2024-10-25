_base_ = '../dcn/cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
