_base_ = ['co_dino_5scale_swin_l_16xb1_1x_coco.py']
# model settings
model = dict(backbone=dict(drop_path_rate=0.6))
max_epochs = 36

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
