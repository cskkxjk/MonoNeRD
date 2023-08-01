# Depth Loss Head for stereo matching supervision.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import torchvision.transforms as T
import cv2
import time
from mononerd.utils.ssim import SSIM
from mononerd.utils.edge_aware_loss import edge_aware_loss_v2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().detach().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

class DepthLossHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_nerf = model_cfg.USE_NERF
        self.valid_mask_threshold = model_cfg.VALID_MASK_THRESHOLD
        self.depth_loss_type = model_cfg.DEPTH_LOSS_TYPE
        self.rgb_loss_type = model_cfg.RGB_LOSS_TYPE
        self.depth_weights = model_cfg.DEPTH_WEIGHTS
        self.rgb_weights = model_cfg.RGB_WEIGHTS
        self.point_cloud_range = point_cloud_range
        self.min_depth = point_cloud_range[0]
        self.max_depth = point_cloud_range[3]
        self.ssim = SSIM(size_average=True).cuda()
        self.forward_ret_dict = {}

    def get_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}

        nerf_left_rgb_loss = torch.tensor(0.).cuda()
        nerf_left_depth_loss = torch.tensor(0.).cuda()
        nerf_right_rgb_loss = torch.tensor(0.).cuda()
        nerf_right_depth_loss = torch.tensor(0.).cuda()
        sdf_loss = torch.tensor(0.).cuda()

        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda()
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda()
        if 'bev_density_vis' in batch_dict:
            tb_dict['bev_density_vis'] = visualize_depth(batch_dict['bev_density_vis'][0][0, 0, :, :])[None, :, :, :]

        # sdf loss
        if 'sdf_near_surface'in batch_dict:
            sdf_near_surface = batch_dict['sdf_near_surface']
            sdf_loss += (sdf_near_surface ** 2).mean()

        # nerf depth loss
        if 'nerf_depth_preds' in batch_dict:
            vis_nerf_depth = visualize_depth(batch_dict['nerf_depth_preds'][0][0, 0, :, :])[None, :, :, :]
            tb_dict['nerf_depth_preds'] = vis_nerf_depth
            gt = batch_dict['depth_gt_img'].squeeze(1)
            mask = (gt > self.min_depth) & (gt < self.max_depth)
            gt = gt[mask]
            nerf_depth_preds = batch_dict['nerf_depth_preds']
            for i, (depth_pred, pred_weight) in enumerate(zip(nerf_depth_preds, self.depth_weights)):
                depth_pred = depth_pred.squeeze(1)[mask]
                # depth_cost = depth_cost.permute(0, 2, 3, 1)[mask]

                for loss_type, loss_type_weight in self.depth_loss_type.items():
                    if depth_pred.shape[0] == 0:
                        print('no gt warning')
                        loss = nerf_depth_preds[i].mean() * 0.0
                    else:
                        if loss_type == "l1":
                            loss = F.smooth_l1_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        elif loss_type == "purel1":
                            loss = F.l1_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        elif loss_type == 'mse':
                            loss = F.mse_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        else:
                            raise NotImplementedError

                    nerf_left_depth_loss += pred_weight * loss_type_weight * loss

        if 'nerf_right_depth_preds' in batch_dict:
            vis_nerf_right_depth = visualize_depth(batch_dict['nerf_right_depth_preds'][0][0, 0, :, :])[None, :, :, :]
            tb_dict['nerf_right_depth_preds'] = vis_nerf_right_depth
            gt = batch_dict['depth_gt_right_img'].squeeze(1)
            mask = (gt > self.min_depth) & (gt < self.max_depth)
            gt = gt[mask]
            nerf_depth_preds = batch_dict['nerf_right_depth_preds']
            for i, (depth_pred, pred_weight) in enumerate(zip(nerf_depth_preds, self.depth_weights)):
                depth_pred = depth_pred.squeeze(1)[mask]
                # depth_cost = depth_cost.permute(0, 2, 3, 1)[mask]

                for loss_type, loss_type_weight in self.depth_loss_type.items():
                    if depth_pred.shape[0] == 0:
                        print('no gt warning')
                        loss = nerf_depth_preds[i].mean() * 0.0
                    else:
                        if loss_type == "l1":
                            loss = F.smooth_l1_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        elif loss_type == "purel1":
                            loss = F.l1_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        elif loss_type == 'mse':
                            loss = F.mse_loss(depth_pred, gt, reduction='none')
                            loss = loss.mean()
                        else:
                            raise NotImplementedError

                    nerf_right_depth_loss += pred_weight * loss_type_weight * loss
        # nerf rgb loss
        if 'nerf_rgb_preds' in batch_dict:
            gt_rgb = batch_dict['left_img']
            tb_dict['nerf_rgb_preds'] = batch_dict['nerf_rgb_preds'][0]
            rgb_preds = batch_dict['nerf_rgb_preds']
            nerf_depth_preds = batch_dict['nerf_depth_preds']
            # gt_rgb = torch.stack((gt_rgb[:, 0, :, :][mask], gt_rgb[:, 1, :, :][mask], gt_rgb[:, 2, :, :][mask]), dim=1)
            gt_rgb = gt_rgb * rgb_std[None, :, None, None] + rgb_mean[None, :, None, None]
            tb_dict['rgb_gts'] = gt_rgb
            for i, (rgb_pred, depth_pred, pred_weight) in enumerate(zip(rgb_preds, nerf_depth_preds, self.rgb_weights)):
                for loss_type, loss_type_weight in self.rgb_loss_type.items():
                    if rgb_pred.shape[0] == 0:
                        print('no gt warning')
                        loss = rgb_pred[i].mean() * 0.0
                    else:
                        if loss_type == "l1":
                            loss = F.smooth_l1_loss(rgb_pred, gt_rgb, reduction='none')
                            loss = loss.mean()
                        elif loss_type == "ssim":
                            loss = 1 - self.ssim(rgb_pred, gt_rgb)
                            loss = loss.mean()
                        elif loss_type == 'edge':
                            disparity = torch.reciprocal(depth_pred)
                            loss = edge_aware_loss_v2(rgb_pred, disparity)
                        else:
                            raise NotImplementedError
                    tb_dict['loss_left_{}'.format(loss_type)] = loss.item()
                    nerf_left_rgb_loss += pred_weight * loss_type_weight * loss

        if 'nerf_right_rgb_preds' in batch_dict:
            right_gt_rgb = batch_dict['right_img']
            tb_dict['nerf_right_rgb_preds'] = batch_dict['nerf_right_rgb_preds'][0]
            right_rgb_preds = batch_dict['nerf_right_rgb_preds']
            right_nerf_depth_preds = batch_dict['nerf_right_depth_preds']
            right_gt_rgb = right_gt_rgb * rgb_std[None, :, None, None] + rgb_mean[None, :, None, None]
            tb_dict['right_rgb_gts'] = right_gt_rgb
            valids_rl = batch_dict['valids_rl']
            right_valid_mask = torch.ge(valids_rl, self.valid_mask_threshold).float()
            for i, (rgb_pred, depth_pred, pred_weight) in enumerate(zip(right_rgb_preds, right_nerf_depth_preds, self.rgb_weights)):
                for loss_type, loss_type_weight in self.rgb_loss_type.items():
                    if rgb_pred.shape[0] == 0:
                        print('no gt warning')
                        loss = rgb_pred[i].mean() * 0.0
                    else:
                        if loss_type == "l1":
                            loss = F.smooth_l1_loss(rgb_pred, right_gt_rgb, reduction='none')
                            loss = loss * right_valid_mask
                            loss = loss.mean()
                        elif loss_type == "ssim":
                            loss = 1 - self.ssim(rgb_pred, right_gt_rgb)
                            loss = loss.mean()
                        elif loss_type == 'edge':
                            disparity = torch.reciprocal(depth_pred)
                            loss = edge_aware_loss_v2(rgb_pred, disparity)
                        else:
                            raise NotImplementedError
                    tb_dict['loss_right_{}'.format(loss_type)] = loss.item()
                    nerf_right_rgb_loss += pred_weight * loss_type_weight * loss
        return nerf_left_rgb_loss, nerf_left_depth_loss, nerf_right_rgb_loss, nerf_right_depth_loss, sdf_loss, tb_dict

    def forward(self, batch_dict):
        # if not self.use_rgb_density:
        #     return batch_dict
        if 'visualization_3d' in batch_dict:
            if 'bev_density_vis' in batch_dict:
                batch_dict['vis_mem']['vis_bev'] = visualize_depth(batch_dict['bev_density_vis'][0][0, 0, :, :]).permute(1, 2, 0).detach().cpu().numpy()
            return batch_dict
        if 'count_time' in batch_dict:
            batch_dict['finished_time_point'] = time.time()
            return batch_dict
        if not self.training and self.use_nerf:
            # depth_pred = batch_dict['depth_preds'][-1].squeeze(1)
            depth_pred = batch_dict['nerf_depth_preds'][-1].squeeze(1)
            # depth_cost = batch_dict['depth_volumes'][0].permute(0, 2, 3, 1)
            # depth_sample = batch_dict['depth_samples']
            gt = batch_dict['depth_gt_img'].squeeze(1)

            mask = (gt > self.min_depth) & (gt < self.max_depth)
            # depth_interval = depth_sample[1] - depth_sample[0]
            assert mask.sum() > 0

            # abs error
            error_map = torch.abs(depth_pred - gt) * mask.float()
            batch_dict['depth_error_map'] = error_map

            # mean_error = error_map[mask].mean()
            median_error = error_map[mask].median()

            # batch_dict['depth_error_local_mean'] = mean_error
            batch_dict['depth_error_all_local_median'] = median_error
            for thresh in [0.2, 0.4, 0.8, 1.6]:
                batch_dict[f"depth_error_all_local_{thresh:.1f}m"] = (error_map[mask] > thresh).float().mean()

            if 'depth_fgmask_img' in batch_dict:
                fg_mask = (gt > self.min_depth) & (gt < self.max_depth) & (batch_dict['depth_fgmask_img'].squeeze(1) > 0)
                local_errs = torch.abs(depth_pred - gt)
                fg_local_errs = local_errs[fg_mask]

                # fg local depth errors per instance
                fg_gts = gt[fg_mask]
                batch_dict['depth_error_fg_local_statistics_perbox'] = []
                fg_ids = batch_dict['depth_fgmask_img'].squeeze(1)[fg_mask].int() - 1
                if len(fg_ids) > 0:
                    for idx in range(fg_ids.min().item(), fg_ids.max().item() + 1):
                        if batch_dict['gt_index'][0][idx] < 0:
                            continue
                        if torch.sum(fg_ids == idx) <= 5:
                            continue
                        errs_i = fg_local_errs[fg_ids == idx]
                        fg_gt_i_median = fg_gts[fg_ids == idx].median().item()
                        num_points_i = (fg_ids == idx).sum().item()
                        batch_dict['depth_error_fg_local_statistics_perbox'].append(dict(
                            distance=fg_gt_i_median,
                            err_median=errs_i.median().item(),
                            num_points=num_points_i,
                            name=batch_dict['gt_names'][0][idx],
                            truncated=batch_dict['gt_truncated'][0][idx],
                            occluded=batch_dict['gt_occluded'][0][idx],
                            difficulty=batch_dict['gt_difficulty'][0][idx],
                            index=batch_dict['gt_index'][0][idx],
                            idx=idx,
                            image_idx=batch_dict['image_idx'][0]
                        ))

                        for thresh in [0.2, 0.4, 0.8, 1.6]:
                            batch_dict['depth_error_fg_local_statistics_perbox'][-1][f"err_{thresh:.1f}m"] = (errs_i > thresh).float().mean().item()

        return batch_dict