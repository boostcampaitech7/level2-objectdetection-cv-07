import json
from mmcv import Config
from config_utils import load_config, update_config
from train_utils import build_and_train_model

# JSON 파일에서 설정값 불러오기
with open('mmdetection/base_config.json', 'r') as f:
    config = json.load(f)

for curr_fold in range(config['k_folds']):
    # Load and update config
    cfg = load_config(config['config_file_path'])
    cfg = update_config(cfg, 
                        config['dataset_root'], 
                        config['split_dataset_root'], 
                        config['work_dir'],
                        config['classes'], 
                        config['k_folds'], 
                        curr_fold)

    # Build and train model
    build_and_train_model(cfg)