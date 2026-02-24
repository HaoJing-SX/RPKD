import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadRP(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        # print("@@@@@@@@@@@@ num_class:", num_class)  # 1
        # print("@@@@@@@@@@@@ input_channels:", input_channels) # 32
        self.rp_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        if self.model_cfg.get('UnMatched_RP_Mask', None):
            point_rp_loss, tb_dict_1 = self.get_matched_rp_loss()
        else:
            point_rp_loss, tb_dict_1 = self.get_rp_loss()

        point_loss = point_rp_loss
        tb_dict.update(tb_dict_1)

        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """

        # print('@@@@@@@@@@@ model_cfg:', self.model_cfg)
        if self.model_cfg.get('USE_RP_FEATURES', False):
            # print('@@@@@@@@@@@ done1')
            point_features = batch_dict['point_features_rp']
        else:
            # print('@@@@@@@@@@@ done3')
            point_features = batch_dict['points']
        # print('@@@@@@@@@@@ point_features_size:', point_features.shape) # [n, 32]

        point_rp_preds = self.rp_layers(point_features)  # (total_points, 1)
        point_rp_preds = torch.sigmoid(point_rp_preds).squeeze(1)
        # print('@@@@@@@@@@@ point_rp_preds_sum:', point_rp_preds.sum())
        if self.model_cfg.get('UnMatched_RP_Mask', None):
            matched_rp_mask = torch.ones([point_rp_preds.shape[0]], dtype=torch.float32).cuda()
            matched_rp_mask *= (batch_dict['ptsv_indexes'] >= 0)
            point_rp_preds = point_rp_preds * matched_rp_mask # unmatched to zero
            # print("@@@@@@@@@@@@ um_rp_mask_size:", matched_rp_mask.shape)
            # print("@@@@@@@@@@@@ um_rp_mask_sum:", matched_rp_mask.sum())
            # print('@@@@@@@@@@@ point_rp_preds_sum2:', point_rp_preds.sum())

        ret_dict = {
            'point_rp_preds': point_rp_preds,
        }

        if self.training:
            point_mfe_rp_labels = batch_dict['point_mfe_rp_labels']
            # print('@@@@@@@@@@@ point_mfe_rp_labels_sum:', point_mfe_rp_labels.sum())
            if self.model_cfg.get('UnMatched_RP_Mask', None):
                unmatched_rp_mask = (torch.ones([point_rp_preds.shape[0]], dtype=torch.float32) * (-1)).cuda()
                unmatched_rp_mask *= (batch_dict['ptsv_indexes'] == -1)
                point_mfe_rp_labels = point_mfe_rp_labels + unmatched_rp_mask # unmatched label to <0
                # print('@@@@@@@@@@@ unmatched_rp_mask_sum:', unmatched_rp_mask.sum())
                # print('@@@@@@@@@@@ point_mfe_rp_labels_sum2:', point_mfe_rp_labels.sum())
            ret_dict['point_rp_labels'] = point_mfe_rp_labels

        self.forward_ret_dict = ret_dict

        # print("@@@@@@@@@@@@ points1:", batch_dict['points'])
        # print("@@@@@@@@@@@@ points1:", batch_dict['points'].shape)
        # print("@@@@@@@@@@@@ point_rp_preds:", point_rp_preds.shape)
        batch_dict['point_rp_preds'] = point_rp_preds
        batch_dict['points'][:, 4] = point_rp_preds
        # print("@@@@@@@@@@@@ points2:", batch_dict['points'])
        # print("@@@@@@@@@@@@ points_size:", batch_dict['points'].size())
        # print("@@@@@@@@@@@@ batch_dict:", batch_dict)

        # a = {}
        # print("stop:", a[10])
        return batch_dict
