_base_ = [
    '/data/ephemeral/home/Jihwan/code/baseline/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/data/ephemeral/home/Jihwan/code/baseline/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/data/ephemeral/home/Jihwan/code/baseline/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '/data/ephemeral/home/Jihwan/code/baseline/mmdetection/configs/_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

model = dict(
    _delete_=True,  # 기존 backbone 설정을 완전히 삭제
    type='FasterRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

# 데이터셋 설정
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/data/ephemeral/home/dataset/train.json',
        img_prefix='/data/ephemeral/home/dataset/train',
        classes=("General trash", "Paper", "Paper pack", "Metal", "Glass", 
                 "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    ),
    val=dict(
        type='CocoDataset',
        ann_file='/data/ephemeral/home/dataset/train.json',
        img_prefix='/data/ephemeral/home/dataset/train',
        classes=("General trash", "Paper", "Paper pack", "Metal", "Glass", 
                 "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    ),
    test=dict(
        type='CocoDataset',
        ann_file='/data/ephemeral/home/dataset/test.json',
        img_prefix='/data/ephemeral/home/dataset/test',
        classes=("General trash", "Paper", "Paper pack", "Metal", "Glass", 
                 "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    )
)

# 옵티마이저 설정
optimizer = dict(
    _delete_=True,
    type='AdamW', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 학습 스케줄 설정
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 체크포인트 설정
checkpoint_config = dict(interval=1)

# 로그 설정
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# 평가 설정
evaluation = dict(interval=1, metric='bbox')