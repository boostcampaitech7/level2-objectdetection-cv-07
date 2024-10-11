from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset

def build_and_train_model(cfg):
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build model and initialize weights
    model = build_detector(cfg.model)
    model.init_weights()
    
    # Train the model
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)