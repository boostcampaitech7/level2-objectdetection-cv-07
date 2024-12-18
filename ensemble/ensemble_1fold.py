from ensemble import ensemble 
import os
import numpy as np

if __name__ == '__main__':
    # TTA.py로 생성된 폴드 별 파일의 앙상블
    # TTA가 inference에서 적용되는 co-dino 모델은 안써도 됨
    # 예시: 
    # 입력파일: deta-swin-large_0_img_size_720_identity.csv, deta-swin-large_0_img_size_720_hflip.csv
    # 출력파일: deta-swin-large_0_img_size_720_identity_hflip_weighted_boxes_fusion_thres_0.7.csv

    conf = {}
    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    
    submission_file_formats = ['deta-swin-large_0_img_size_720_{transform}.csv', 
                               'deta-swin-large_1_img_size_720_{transform}.csv',
                               'deta-swin-large_2_img_size_680_{transform}.csv',
                               'deta-swin-large_3_img_size_720_{transform}.csv',
                               'deta-swin-large_4_img_size_720_{transform}.csv',]

    conf['mode'] = 'test'
    conf['transforms'] = ['identity', 'hflip']
    conf['prefix'] = '1022'
    conf['ensemble-fold'] = 1
    conf['algorithms'] = ['nms', 'weighted_boxes_fusion']
    conf['iou_thresholds'] = np.arange(0.3, 0.91, 0.1).tolist()
    conf['annotation_path'] = '../../../../data/dataset/test.json'

    conf['submission_fold_path'] = os.path.join(py_dir_path, f'../TTA/result/TTA/{conf["mode"]}')
    conf['output_fold_path'] = os.path.join(py_dir_path, f'result/ensemble-{conf["ensemble-fold"]}fold/{conf["mode"]}')
    

    for submission_file_format in submission_file_formats:
        conf['submission_file_format'] = submission_file_format
        conf['output_name_format'] = conf['prefix'] + '_' + conf['submission_file_format'][0:30] + '_' +  '_'.join(conf['transforms']) + '_{algorithm}_thres_{iou_threshold:.1f}.csv'
        conf['output_path_format'] = conf['output_fold_path'] + '/' + conf['output_name_format']
        
        ensemble(conf)



