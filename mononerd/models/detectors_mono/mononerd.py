from mononerd.ops.iou3d_nms import iou3d_nms_utils
from .stereo_detector3d_template import StereoDetector3DTemplate
import torch
import torch.nn as nn
from torch.autograd import Variable

class MonoNeRD(StereoDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.loss_cfg = model_cfg.LOSS_CFG
        self.loss_weights = self.loss_cfg.weights
        self.weights = torch.zeros(5).cuda()
        if self.loss_cfg.left_depth:
            self.weights[0] = torch.exp(self.weights[0]) * self.loss_weights[0]
        if self.loss_cfg.left_rgb:
            self.weights[1] = torch.exp(self.weights[1]) * self.loss_weights[1]
        if self.loss_cfg.right_depth:
            self.weights[2] = torch.exp(self.weights[2]) * self.loss_weights[2]
        if self.loss_cfg.right_rgb:
            self.weights[3] = torch.exp(self.weights[3]) * self.loss_weights[3]
        if self.loss_cfg.sdf:
            self.weights[4] = torch.exp(self.weights[4]) * self.loss_weights[4]

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, ret_dicts = self.post_processing(batch_dict)
            for k in batch_dict.keys():
                if k.startswith('depth_error_'):
                    if isinstance(batch_dict[k], list):
                        ret_dicts[k] = batch_dict[k]
                    elif len(batch_dict[k].shape) == 0:
                        ret_dicts[k] = batch_dict[k].item()

            if getattr(self, 'dense_head_2d', None) and 'boxes_2d_pred' in batch_dict:
                assert len(pred_dicts) == len(batch_dict['boxes_2d_pred'])
                for pred_dict, pred_2d_dict in zip(pred_dicts, batch_dict['boxes_2d_pred']):
                    pred_dict['pred_boxes_2d'] = pred_2d_dict['pred_boxes_2d']
                    pred_dict['pred_scores_2d'] = pred_2d_dict['pred_scores_2d']
                    pred_dict['pred_labels_2d'] = pred_2d_dict['pred_labels_2d']
            pred_dicts[0]['batch_dict'] = batch_dict

            return pred_dicts, ret_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_nerf_left_rgb, loss_nerf_left_depth, loss_nerf_right_rgb, loss_nerf_right_depth, loss_sdf, tb_dict = self.depth_loss_head.get_loss(batch_dict, tb_dict)
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_nerf_left_depth': loss_nerf_left_depth.item(),
            'loss_nerf_left_rgb': loss_nerf_left_rgb.item(),
            'loss_nerf_right_depth': loss_nerf_right_depth.item(),
            'loss_nerf_right_rgb': loss_nerf_right_rgb.item(),
            'loss_sdf': loss_sdf.item(),
            **tb_dict
        }
        loss = loss_rpn + \
               loss_nerf_left_depth * self.weights[0] + \
               loss_nerf_left_rgb * self.weights[1] + \
               loss_nerf_right_depth * self.weights[2] + \
               loss_nerf_right_rgb * self.weights[3] + \
               loss_sdf * self.weights[4]

        if getattr(self, 'dense_head_2d', None):
            loss_rpn_2d, tb_dict = self.dense_head_2d.get_loss(batch_dict, tb_dict)
            tb_dict['loss_rpn2d'] = loss_rpn_2d.item()
            loss += loss_rpn_2d

        return loss, tb_dict, disp_dict

    def get_iou_map(self, batch_dict):
        batch_size = batch_dict['batch_size']
        iou_map_results = []

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]  # [N_anchors, 7]
            gt_boxes = batch_dict['gt_boxes'][index]

            if gt_boxes.shape[0] <= 0:
                iou_map_results.append(None)
            else:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(
                    box_preds[:, 0:7], gt_boxes[:, 0:7])
                iou_map_results.append(iou3d_roi.detach().cpu().numpy())

        return iou_map_results
