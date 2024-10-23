from .cascade_mask_rcnn_mvitv2_b_3x import model, optimizer, train, lr_multiplier
from .common.coco_loader_lsj import dataloader


model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.depth = 80
model.backbone.bottom_up.num_heads = 3
model.backbone.bottom_up.last_block_indexes = (3, 11, 71, 79)
model.backbone.bottom_up.drop_path_rate = 0.6
model.backbone.bottom_up.use_act_checkpoint = True

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_H_in21k.pyth"


# trash image data에 맞게 클래스 개수 수정
model.roi_heads.num_classes = 10

# 데이터셋 등록
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# MetadataCatalog는 메타데이터를 설정
MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False
dataloader.train.dataset.names='coco_trash_train'
dataloader.test.dataset.names='coco_trash_test'
dataloader.evaluator.dataset_name='coco_trash_test'


# 훈련 이터레이션 수 설정
train.max_iter = 20000

# 체크포인트 저장 주기 설정
train.checkpointer.period = 1000
train.checkpointer.max_to_keep=5
# 테스트 주기 설정
train.eval_period = 1000

# 필요에 따라 학습률 스케줄러 조정
lr_multiplier.scheduler.milestones = [1, 10000, 15000]
lr_multiplier.scheduler.values=[1.0, 0.1, 0.01]

del model.roi_heads['mask_in_features']
del model.roi_heads['mask_pooler']
del model.roi_heads['mask_head']

### batch size
dataloader.train.total_batch_size=4

