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
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out',type=str,help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument( '--show-dir',
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
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
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

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    #pdb.set_trace()
    # 여기서부터 디버깅 코드 추가
    #print("Test evaluator config:", cfg.test_evaluator)

    #print("Model structure:")
    #print(runner.model)

    #print("Checking test dataloader...")
    #for idx, data in enumerate(runner.test_dataloader):
    #    print(f"Sample {idx}:")
    #    print(data.keys())
    #    if idx > 2:  # 처음 3개 샘플만 확인
    #        break

    #print("Checking model output...")
    #import torch
    #with torch.no_grad():
    #    for idx, data in enumerate(runner.test_dataloader):
    #        result = runner.model.test_step(data)
    #        print(f"Sample {idx} result:", result)
    #        if idx > 2:
    #            break
    
    #pdb.set_trace()

    cfg.test_evaluator['format_only'] = False
    #cfg.test_evaluator['metric'] = ['bbox']

    # start testing
    print("Starting test...")
    output = runner.test()
    #pdb.set_trace()
    print(f"Test completed. Output type: {type(output)}, length: {len(output) if output else 0}")

    metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))
    print("Evaluation metrics:", metrics)

    epoch = '8'

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.test_evaluator.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
    submission.head()


if __name__ == '__main__':
    main()