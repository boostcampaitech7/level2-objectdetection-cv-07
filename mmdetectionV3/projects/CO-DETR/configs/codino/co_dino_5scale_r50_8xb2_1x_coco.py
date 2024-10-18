_base_ = './co_dino_5scale_r50_lsj_8xb2_1x_coco.py'
data_root = '/data/ephemeral/home/dataset/'



model = dict(
    use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(700, 700), (800, 800), (900, 900)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(600, 600),
                    allow_negative_crop=False),
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(600, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
