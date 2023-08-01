import argparse
import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import torch
import time

from vis_utils import vis_utils
from mononerd.config import cfg, cfg_from_list, cfg_from_yaml_file, update_cfg_by_args, log_config_to_file
from mononerd.datasets import build_dataloader
from mononerd.models import build_network
from mononerd.utils import common_utils

torch.backends.cudnn.benchmark = True


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    # basic testing options
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default=None, help='exp path for this experiment')
    parser.add_argument('--vis_tag', type=str, default='sequence_0', help='eval tag for this experiment')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')

    # loading options
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to evaluate')
    parser.add_argument('--ckpt_id', type=int, default=None, help='checkpoint id to evaluate')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    # distributed options
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='pytorch')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # config options
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--trainval', action='store_true', default=False, help='')
    parser.add_argument('--imitation', type=str, default="2d")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    update_cfg_by_args(cfg, args)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    cfg.DATA_CONFIG.VIS_TAG = args.vis_tag
    np.random.seed(1024)

    assert args.ckpt or args.ckpt_id, "pls specify ckpt or ckpt_dir or ckpt_id"

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def vis_single_ckpt(model, vis_loader, args, vis_output_dir, logger, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    vis_utils.visualization_and_save_pickle(
        cfg, model, vis_loader, logger, dist_test=dist_test,
        result_dir=vis_output_dir
    )


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    vis_output_dir = cfg.ROOT_DIR / 'data' / 'kitti'/ 'visualization' / args.vis_tag / 'vis_3d'
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = vis_output_dir / ('log_vis.txt')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES={}'.format(gpu_list))
    logger.info('eval output dir: {}'.format(vis_output_dir))

    if dist_test:
        logger.info('total_batch_size: {}'.format(total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    vis_set, vis_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=vis_set)
    with torch.no_grad():
        vis_single_ckpt(model, vis_loader, args, vis_output_dir, logger, dist_test=dist_test)


if __name__ == '__main__':
    main()
