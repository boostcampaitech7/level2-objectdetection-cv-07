# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import sys
sys.path.append('/data/ephemeral/home/Jeongseon/mmdetection/V3/base')

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

import pdb
import json
from PIL import Image

import pickle

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--skip',
        action='store_true',
        default=False)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    
    cfg.train_dataloader.dataset.data_root = '/data/ephemeral/home/dataset/'
    cfg.train_dataloader.dataset.ann_file = '/data/ephemeral/home/dataset/split/train_0_5.json'
    cfg.train_dataloader.dataset.data_prefix = {'img': '/data/ephemeral/home/dataset/'}

    cfg.val_dataloader.dataset.data_root = '/data/ephemeral/home/dataset/'
    cfg.val_dataloader.dataset.ann_file = '/data/ephemeral/home/dataset/split/valid_0_5.json'
    cfg.val_evaluator.ann_file = '/data/ephemeral/home/dataset/split/valid_0_5.json'
    cfg.val_dataloader.dataset.data_prefix = {'img': '/data/ephemeral/home/dataset/'}

    cfg.test_dataloader.dataset.data_root = '/data/ephemeral/home/dataset/' 
    cfg.test_dataloader.dataset.ann_file = '/data/ephemeral/home/dataset/test.json'
    cfg.test_evaluator.ann_file = '/data/ephemeral/home/dataset/test.json'
    cfg.test_dataloader.dataset.data_prefix = {'img': '/data/ephemeral/home/dataset/'}

    cfg.model.bbox_head.num_classes = 10 
    metainfo = {'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")}
    cfg.test_dataloader.dataset.metainfo = metainfo 

    cfg.debug = True

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        # TTA 파이프라인 설정이 없다면 기본값으로 설정            
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            
            # 기본적으로 좌우 반전 적용하는 설정
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.skip is False:
        # add `DumpResults` dummy metric
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path='results.pkl'))

        # start testing
        test_results = runner.test()
    
    with open('results.pkl', 'rb') as f:
        test_results = pickle.load(f)

    # --- 결과 후처리 및 submission.csv 파일 생성 ---
        # COCO dataset 정보를 이용하여 output을 후처리
        ann_file_path = cfg.test_evaluator.get('ann_file', None)
        if ann_file_path is None:
            raise AttributeError("Annotation file path is missing in the config")
        coco = COCO(ann_file_path)

        img_ids = coco.getImgIds()  # 이미지 ID 가져오기
        class_num = len(coco.getCatIds())  # 클래스 수 정의 (10 등)
        
        prediction_strings = []
        file_names = []

        # test_results를 COCO 형식으로 변환 후 submission 파일 작성
        for i, out in enumerate(test_results):
            prediction_string = ''
            image_info = coco.loadImgs(img_ids[i])[0]  # 이미지 정보 로드

            pred_instance = out['pred_instances']
            for i in range(len(pred_instance['scores'])):
                
                if pred_instance['scores'][i].item() <= 0.05:
                    pass

                label = str(pred_instance['labels'][i].item())
                score = str(pred_instance['scores'][i].item())
                bbox_str = ' '.join(map(str, pred_instance['bboxes'][i].tolist()))
                
                prediction_string += label + ' ' + score + ' ' + bbox_str + ' '

            # 파일 이름과 함께 prediction_string을 저장
            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])

        # 결과를 데이터프레임으로 저장 (submission.csv 파일 생성)
        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names

        # 결과 CSV 저장
        submission.to_csv(os.path.join(cfg.work_dir, 'submission.csv'), index=False)

        print(submission.head())  # 결과 미리보기

if __name__ == '__main__':
    main()