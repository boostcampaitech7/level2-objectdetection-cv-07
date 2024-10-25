from ensemble import ensemble_five
import os
import numpy as np

if __name__ == '__main__':
    # ensemble_1fold.py로 생성된 결과를 5-fold 앙상블 
    # 예시:
    # 입력파일: 1022_deta-swin-large_{i}_img_size_720_identity_hflip_weighted_boxes_fusion_thres_0.7.csv (i=0,1,2,3,4)
    # 출력파일: 1022_deta_5_fold_weighted_boxes_fusion_0.7.csv

    conf = {}
    py_dir_path = os.path.dirname(os.path.abspath(__file__))

    algorithm = 'weighted_boxes_fusion'
    iou_threshold = 0.7
    conf["mode"] = 'test'
    
    conf['submission_file_suffix'] = f'identity_hflip_{algorithm}_thres_{iou_threshold}.csv'
    conf['submission_file_prefix'] = ['1022_deta-swin-large_0_img_size_720',
                                    '1022_deta-swin-large_1_img_size_720',
                                    '1022_deta-swin-large_2_img_size_680',
                                    '1022_deta-swin-large_3_img_size_720',
                                    '1022_deta-swin-large_4_img_size_720']
    
    conf['submission_file_paths'] = []
    for prefix in conf['submission_file_prefix']:
        submission_file_name = prefix + '_' + conf['submission_file_suffix']
        submission_file_path = os.path.join(py_dir_path + '/result/ensemble-1fold/test', submission_file_name)
        conf['submission_file_paths'].append(submission_file_path)

    
    conf['transforms'] = ['identity', 'hflip']
    conf['prefix'] = '1022'
    conf['ensemble-fold'] = 5
    conf['algorithms'] = ['nms', 'weighted_boxes_fusion']
    conf['iou_thresholds'] = np.arange(0.3, 0.91, 0.1).tolist()
    conf['annotation_path'] = '/home/jin/project/Object detection/data/dataset/test.json'

    conf['submission_fold_path'] = os.path.join(py_dir_path, f'result/TTA/{conf["mode"]}')
    conf['output_fold_path'] = os.path.join(py_dir_path, f'result/ensemble-{conf["ensemble-fold"]}fold/{conf["mode"]}')
    
    conf['output_name_format'] = conf['prefix'] + f'_{conf["ensemble-fold"]}-fold_' +  '_'.join(conf["transforms"]) + f'_{algorithm}_thres_{iou_threshold:.1f}.csv'
    conf['output_path_format'] = conf['output_fold_path'] + '/' + conf['output_name_format']
        
    ensemble_five(conf)



