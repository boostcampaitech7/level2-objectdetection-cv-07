# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    # add parser ; data metainfo
    parser.add_argument('--classes', type=str, default='General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing')
    parser.add_argument('--data_root', type=str, default='/data/ephemeral/home/dataset/')
    parser.add_argument('--ann_file', type=str, default='/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/train_0_5.json') 
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--work_dir', type=str, default='/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Backbone/Sojeong/co-dino/mmdetection/work_dirs/co_dino')
    parser.add_argument('--eval_ann_file', type=str, default='/data/ephemeral/home/Sojeong/level2-objectdetection-cv-07/Split_data/valid_0_5.json')
    parser.add_argument('--test_ann_file', type=str, default='/data/ephemeral/home/dataset/test.json')
    parser.add_argument('--image_size', type=int, nargs=2, default=(800, 800))
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
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

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    metainfo = {'classes': args.classes.split(', ')}
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.val_dataloader.dataset.metainfo = metainfo
    
    cfg.image_size = args.image_size
    cfg.work_dir = args.work_dir
    
    cfg.model.bbox_head[0].num_classes = args.num_classes
    cfg.model.query_head.num_classes = args.num_classes
    cfg.model.roi_head[0].bbox_head.num_classes = args.num_classes
    
    cfg.train_dataloader.dataset.data_root = args.data_root
    cfg.train_dataloader.dataset.ann_file = args.ann_file
    cfg.train_dataloader.dataset.data_prefix=dict(img='')
        
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.val_dataloader.dataset.ann_file = args.eval_ann_file
    cfg.val_evaluator.ann_file = args.eval_ann_file
    cfg.val_dataloader.dataset.data_prefix=dict(img='')
    
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.ann_file = args.test_ann_file
    cfg.test_evaluator.ann_file = args.test_ann_file
    cfg.test_dataloader.dataset.data_prefix=dict(img='')
    
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.batch_size = args.batch_size
    
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.val_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.num_workers = args.num_workers
    #import IPython; IPython.embed(colors='Linux');exit(1);
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()