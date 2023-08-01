import pickle
import time

import numpy as np
import torch
import tqdm

from mononerd.models import load_data_to_gpu
from mononerd.utils import common_utils

def visualization_and_save_pickle(cfg, model, dataloader, logger, dist_test=False, result_dir=None):
    logger.info('*************** Start generate 3D visualization data ******************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    # start_time = time.time()
    backbone_latency = 0.
    ours_latency = 0.
    detection_latency = 0.
    visualization_3d = False
    count_time = True
    for i, batch_dict in enumerate(dataloader):
        if visualization_3d:
            batch_dict['visualization_3d'] = visualization_3d
        if count_time:
            batch_dict['count_time'] = count_time
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        if 'count_time' in pred_dicts[0]['batch_dict']:
            backbone_latency += pred_dicts[0]['batch_dict']['2d_backbone_end'] - pred_dicts[0]['batch_dict']['2d_backbone_start']
            ours_latency += (pred_dicts[0]['batch_dict']['our_end_0'] - pred_dicts[0]['batch_dict']['our_start_0']) + \
                            (pred_dicts[0]['batch_dict']['our_end_1'] - pred_dicts[0]['batch_dict']['our_start_1'])
            detection_latency += pred_dicts[0]['batch_dict']['finished_time_point'] - pred_dicts[0]['batch_dict']['our_end_1']
        if result_dir and 'visualization_3d' in pred_dicts[0]['batch_dict']:
            vis_mem = pred_dicts[0]['batch_dict']['vis_mem']
            file_name = str(vis_mem['idx']) + '.pkl'
            file_path = result_dir / file_name
            with open(file_path, 'wb') as handle:
                pickle.dump(vis_mem, handle)
                print("wrote to", file_path)
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    sec_backbone_per_example = backbone_latency / len(dataloader.dataset)
    sec_ours_per_example = ours_latency / len(dataloader.dataset)
    sec_detection_per_example = detection_latency / len(dataloader.dataset)
    logger.info('sec_backbone_per_example: %.4f second).' % sec_backbone_per_example)
    logger.info('sec_ours_per_example: %.4f second).' % sec_ours_per_example)
    logger.info('sec_detection_per_example: %.4f second).' % sec_detection_per_example)
    if result_dir and 'visualization_3d' in pred_dicts[0]['batch_dict']:
        logger.info('Result is save to %s' % result_dir)
    else:
        logger.info('Nothing to save')
    logger.info('****************Generation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
