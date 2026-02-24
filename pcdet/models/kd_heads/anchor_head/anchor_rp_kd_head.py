import torch

from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.models.dense_heads.point_head_template import PointHeadTemplate
from pcdet.utils import loss_utils, box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# from pcdet.models.dense_heads.anchor_head_template import AnchorHeadTemplate

class AnchorRPKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super(AnchorRPKDHead, self).__init__(model_cfg, dense_head)

    def build_rp_kd_loss(self):
        # rp_kd loss
        if self.model_cfg.KD_LOSS.RP_LOSS.type in ['MSELoss']:
            # MSELoss
            self.kd_rp_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.HM_LOSS.type)(reduction='none')
        else:
            raise NotImplementedError
        # direction loss
        # self.kd_dir_loss_func = loss_utils.WeightedCrossEntropyLoss()

    def get_rp_kd_loss(self, batch_dict, tb_dict):
        loss_cfg = self.model_cfg.KD_LOSS
        if self.model_cfg.RP_KD.get('MMDv2', False):
            rp_pred_stu = batch_dict['point_rp_preds']
            sp_rp_pred_tea = batch_dict['sp_rp_pred_tea']
            ptsp_indexes = batch_dict['ptsp_indexes']
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ rp_pred_stu_size:', rp_pred_stu.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ sp_rp_pred_tea_size:', sp_rp_pred_tea.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ ptsp_indexes_size:', ptsp_indexes.shape)
            batch_size = batch_dict['batch_size']
            points = batch_dict['points'][:, :4]
            batch_point_index = batch_dict['points'][:, 0]
            batch_spoint_index = batch_dict['s_points2'][:, 0]
            mid_rp_pred_stu = []
            mid_points = []
            mid_rp_pred_tea = []
            for k in range(batch_size):
                bs_point_mask = (batch_point_index == k)
                now_ptsp_indexes = ptsp_indexes[bs_point_mask].long()
                now_rp_pred_stu = rp_pred_stu[bs_point_mask]
                now_rp_pred_stu = now_rp_pred_stu[now_ptsp_indexes >= 0]  # remove unmatched compress point's preds
                now_points = points[bs_point_mask]
                now_points = now_points[now_ptsp_indexes >= 0]  # remove unmatched compress point's preds
                bs_spoint_mask = (batch_spoint_index ==k)
                now_sp_rp_pred_tea = sp_rp_pred_tea[bs_spoint_mask]
                now_ptsp_indexes = now_ptsp_indexes[now_ptsp_indexes >= 0]
                now_rp_pred_tea = now_sp_rp_pred_tea[now_ptsp_indexes]
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_ptsp_indexes_size:', now_ptsp_indexes.shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_rp_pred_tea_size:', now_rp_pred_tea.shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_rp_pred_stu_size:', now_rp_pred_stu.shape)
                mid_rp_pred_stu.append(now_rp_pred_stu)
                mid_points.append(now_points)
                mid_rp_pred_tea.append(now_rp_pred_tea)
        else:
            rp_pred_stu = batch_dict['point_rp_preds']
            sv_rp_pred_tea = batch_dict['sv_rp_pred_tea']
            ptsv_indexes = batch_dict['ptsv_indexes']
            points = batch_dict['points'][:, :4]
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ rp_pred_stu_size:', rp_pred_stu.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ sv_rp_pred_tea_size:', sv_rp_pred_tea.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ ptsv_indexes_size:', ptsv_indexes.shape)
            batch_size = batch_dict['batch_size']
            batch_point_index = batch_dict['points'][:, 0]
            batch_voxel_index = batch_dict['voxel_coords_tea2'][:, 0]
            mid_rp_pred_stu = []
            mid_points = []
            mid_rp_pred_tea = []
            for k in range(batch_size):
                bs_point_mask = (batch_point_index == k)
                now_ptsv_indexes = ptsv_indexes[bs_point_mask].long()
                now_rp_pred_stu = rp_pred_stu[bs_point_mask]
                now_rp_pred_stu = now_rp_pred_stu[now_ptsv_indexes >= 0] # remove unmatched compress point's preds
                now_points = points[bs_point_mask]
                now_points = now_points[now_ptsv_indexes >= 0] # remove unmatched compress point's preds
                now_ptsv_indexes = now_ptsv_indexes[now_ptsv_indexes >= 0] # remove unmatched compress point's indexes
                bs_voxel_mask = (batch_voxel_index == k)
                now_sv_rp_pred_tea = sv_rp_pred_tea[bs_voxel_mask]
                now_rp_pred_tea = now_sv_rp_pred_tea[now_ptsv_indexes]
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_ptsv_indexes_size:', now_ptsv_indexes.shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_sv_rp_pred_tea _size:', now_sv_rp_pred_tea .shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_rp_pred_tea_size:', now_rp_pred_tea.shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@ now_rp_pred_stu_size:', now_rp_pred_stu.shape)
                mid_rp_pred_stu.append(now_rp_pred_stu)
                mid_points.append(now_points)
                mid_rp_pred_tea.append(now_rp_pred_tea)

        rp_pred_stu = torch.cat(mid_rp_pred_stu, dim=0)
        points = torch.cat(mid_points, dim=0)
        rp_pred_tea = torch.cat(mid_rp_pred_tea, dim=0)

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@ rp_pred_tea_size:', rp_pred_tea.shape)
        kd_rp_loss = 0
        assert rp_pred_stu.shape == rp_pred_tea.shape

        if loss_cfg.RP_LOSS.weight == 0:
            kd_rp_loss_raw = 0
        elif loss_cfg.RP_LOSS.type in ['MSELoss']:
            kd_rp_loss_all = self.kd_rp_loss_func(rp_pred_stu, rp_pred_tea)
            mask = torch.ones([rp_pred_stu.shape[0]], dtype=torch.float32).cuda()
            if loss_cfg.RP_LOSS.get('point_fg_mask', None):
                # for point_fg
                gt_boxes = batch_dict['gt_boxes']
                # print("@@@@@@@@@@@@@@@@@ points_size:", points.shape)
                # print("@@@@@@@@@@@@@@@@@ mask_size:", mask.shape)
                assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
                assert points.shape.__len__() in [2], 'points.shape=%s' % str(points.shape)
                batch_size = gt_boxes.shape[0]
                extend_gt_boxes = box_utils.enlarge_box3d(
                    gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.2, 0.2, 0.2]
                ).view(batch_size, -1, gt_boxes.shape[-1])
                bs_idx = points[:, 0]
                point_cls_labels = points.new_zeros(points.shape[0]).long()
                for k in range(batch_size):
                    bs_mask = (bs_idx == k)
                    points_single = points[bs_mask][:, 1:4]
                    point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    box_fg_flag = (box_idxs_of_pts >= 0)
                    extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    fg_flag = box_fg_flag
                    ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                    point_cls_labels_single[ignore_flag] = -1
                    gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
                    point_cls_labels_single[fg_flag] = gt_box_of_fg_points[:, -1].long()
                    point_cls_labels[bs_mask] = point_cls_labels_single
                mask *= (point_cls_labels > 0)
                # print("@@@@@@@@@@@@@@@@@ mask_sum:", mask.sum())
                # print("@@@@@@@@@@@@@@@@@ mask_size:", mask.shape)
            kd_rp_loss_raw = (kd_rp_loss_all * mask).sum() / (mask.sum() + 1e-6)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ kd_rp_loss_all_size:', kd_rp_loss_all.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ mask_size:', mask.shape)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@ kd_rp_loss_raw_size:', kd_rp_loss_raw)
        else:
            raise NotImplementedError

        kd_rp_loss += loss_cfg.RP_LOSS.weight * kd_rp_loss_raw
        # print("@@@@@@@@@@@@@@@@@ kd_rp_loss:", kd_rp_loss)

        # a = {}
        # print("stop:", a[10])

        tb_dict['kd_rp_ls'] = kd_rp_loss if isinstance(kd_rp_loss, float) else kd_rp_loss.item()

        kd_loss = kd_rp_loss

        return kd_loss, tb_dict

    def cal_kd_reg_loss(self, box_preds_stu, box_preds_tea):
        box_cls_labels = self.dense_head.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds_stu.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # print("@@@@@@@@@@ box_preds_stu_size:", box_preds_stu.shape) # [b, 200, 176, 42]
        # print("@@@@@@@@@@ box_preds_tea_size:", box_preds_tea.shape) # [b, 200, 176, 42]
        box_preds_stu = box_preds_stu.view(batch_size, -1,
                                           box_preds_stu.shape[-1] // self.dense_head.num_anchors_per_location)
        box_preds_tea = box_preds_tea.view(batch_size, -1,
                                           box_preds_tea.shape[-1] // self.dense_head.num_anchors_per_location)
        # print("@@@@@@@@@@ box_preds_stu_size:", box_preds_stu.shape) # [b, 211200, 7]
        # print("@@@@@@@@@@ box_preds_tea_size:", box_preds_tea.shape) # [b, 211200, 7]
        # sin(a - b) = sinacosb-cosasinb
        box_preds_stu_sin, box_preds_tea_sin = self.add_sin_difference(box_preds_stu, box_preds_tea)
        kd_reg_loss_raw = self.kd_reg_loss_func(box_preds_stu_sin, box_preds_tea_sin, weights=reg_weights)  # [N, M]
        # print("@@@@@@@@@@ box_preds_stu_sin_size:", box_preds_stu_sin.shape) # [b, 211200, 7]
        # print("@@@@@@@@@@ box_preds_tea_sin_size:", box_preds_tea_sin.shape) # [b, 211200, 7]
        # print("@@@@@@@@@@ reg_weights_size:", reg_weights.shape) # [b, 211200]
        # print("@@@@@@@@@@ kd_reg_loss_raw_size:", kd_reg_loss_raw.shape) # # [b, 211200, 7]
        kd_reg_loss_raw = kd_reg_loss_raw.sum() / batch_size

        return kd_reg_loss_raw

