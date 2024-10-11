import json
from inference_utils import get_cfg, build_data, load_model, inference

# JSON 파일에서 설정값 불러오기
with open('base_config.json', 'r') as f:
    config = json.load(f)
    
# 전체 inference 파이프라인 실행 함수
def run_inference(config_file_path, work_dir, dataset_root, classes, epoch):
    cfg = get_cfg(config_file_path, work_dir, dataset_root, classes, epoch)
    dataset, data_loader = build_data(cfg)
    model = load_model(cfg, dataset, epoch)
    result_file = inference(model, data_loader, cfg, epoch)
    print(f"Inference complete. Results saved to {result_file}")

for curr_fold in range(config['k_folds']):
    if __name__ == '__main__':
        epoch = 'latest'
        # Inference 실행
        run_inference(config['config_file_path'], 
                      f'{config['work_dir']}/fold_{curr_fold},
                      config['dataset_root'], 
                      config['classes'],
                      epoch)