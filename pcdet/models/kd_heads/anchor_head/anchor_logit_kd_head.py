import torch

from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.utils import loss_utils

# from pcdet.models.dense_heads.anchor_head_template import AnchorHeadTemplate

class AnchorLogitKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super(AnchorLogitKDHead, self).__init__(model_cfg, dense_head)

    def build_logit_kd_loss(self):
        # logit kd cls loss
        if self.model_cfg.KD_LOSS.HM_LOSS.type in ['FocalLoss']:
            self.kd_hm_loss_func = loss_utils.ConsisSigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        elif self.model_cfg.KD_LOSS.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            # MSELoss
            self.kd_hm_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.HM_LOSS.type)(reduction='none')
            # print('@@@@@@@@@@@@@@ self.kd_hm_loss_func_logit:', self.kd_hm_loss_func)
        else:
            raise NotImplementedError

        # logit kd regression loss
        if self.model_cfg.KD_LOSS.REG_LOSS.type == 'WeightedSmoothL1Loss':
            self.kd_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.REG_LOSS.type)(
                code_weights=self.model_cfg.KD_LOSS.REG_LOSS.code_weights
            )
        else:
            raise NotImplementedError

        # direction loss
        # self.kd_dir_loss_func = loss_utils.WeightedCrossEntropyLoss()

    def get_logit_kd_loss(self, batch_dict, tb_dict):
        loss_cfg = self.model_cfg.KD_LOSS
        cls_pred_stu = self.dense_head.forward_ret_dict['cls_preds']
        box_pred_stu = self.dense_head.forward_ret_dict['box_preds']
        # dir_pred_stu = self.dense_head.forward_ret_dict['dir_cls_preds']
        # print("@@@@@@@@@@@@@@@@@ cls_pred_stu_size:", cls_pred_stu.shape) # [2, 200, 176, 18]
        # print("@@@@@@@@@@@@@@@@@ box_pred_stu_size:", box_pred_stu.shape) # [2, 200, 176, 42]

        cls_pred_tea = batch_dict['cls_preds_tea']
        box_pred_tea = batch_dict['box_preds_tea']
        # dir_pred_tea = batch_dict['dir_cls_preds_tea']
        # print("@@@@@@@@@@@@@@@@@ cls_pred_tea_size:", cls_pred_tea.shape) # [2, 200, 176, 18]
        # print("@@@@@@@@@@@@@@@@@ box_pred_tea_size:", box_pred_tea.shape) # [2, 200, 176, 42]

        kd_hm_loss = 0
        kd_reg_loss = 0
        # kd_dir_loss = 0

        assert cls_pred_stu.shape == cls_pred_tea.shape

        bs, height, width, n_anchor = cls_pred_stu.shape

        # classification loss
        if loss_cfg.HM_LOSS.weight == 0:
            kd_hm_loss_raw = 0
        elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            if loss_cfg.HM_LOSS.sigmoid:
                # print("@@@@@@@@@@@@@@@@@ done1") # done
                cls_pred_tea = self.sigmoid(cls_pred_tea)
                cls_pred_stu = self.sigmoid(cls_pred_stu)

            kd_hm_loss_all = self.kd_hm_loss_func(cls_pred_stu, cls_pred_tea)
            # print("@@@@@@@@@@@@@@@@@ kd_hm_loss_all_size:", kd_hm_loss_all.shape) # [2, 200, 176, 18]

            mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
            # print("@@@@@@@@@@@@@@@@@ mask.sum():", mask.sum())
            # print("@@@@@@@@@@@@@@@@@ kd_hm_loss_raw:", (kd_hm_loss_all * mask.unsqueeze(3)).sum() / (mask.sum() + 1e-6))

            if loss_cfg.HM_LOSS.get('fg_mask_spatial', None):
                # print("@@@@@@@@@@@@@@@@@ done2") # no
                fg_mask_spatial = self.cal_fg_mask_from_gt_boxes_and_spatial_mask(
                    batch_dict['gt_boxes'], batch_dict['spatial_mask_tea']
                ).float()
                if loss_cfg.HM_LOSS.get('soft_mask', None):
                    fg_mask_spatial *= cls_pred_tea.max(dim=-1)[0]

                mask *= fg_mask_spatial

            if loss_cfg.HM_LOSS.get('fg_mask_anchor', None):
                # print("@@@@@@@@@@@@@@@@@ done3") # done
                # count_ignore
                fg_mask_anchor = self.cal_fg_mask_from_gt_anchors(
                    self.dense_head.forward_ret_dict['box_cls_labels'],
                    anchor_shape=cls_pred_stu.shape,
                    num_anchor=self.dense_head.num_anchors_per_location,
                    count_ignore=loss_cfg.HM_LOSS.count_ignore
                ).float()
                # print("@@@@@@@@@@@@@@@@@ fg_mask_anchor1:", fg_mask_anchor.max())
                if loss_cfg.HM_LOSS.get('soft_mask', None):
                    fg_mask_anchor *= cls_pred_tea.max(dim=-1)[0]
                mask *= fg_mask_anchor

            if loss_cfg.HM_LOSS.get('tea_mask_anchor', None):
                # print("@@@@@@@@@@@@@@@@@ done4") # no
                tea_mask_anchor = (cls_pred_tea.max(dim=-1)[0] > loss_cfg.HM_LOSS.thresh).float()
                mask *= tea_mask_anchor

            # print("@@@@@@@@@@@@@@@@@ mask.unsqueeze(3):", mask.unsqueeze(3).shape) # [2, 200, 176, 1]
            kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(3)).sum() / (mask.sum() + 1e-6)
            # print("@@@@@@@@@@@@@@@@@ mask.sum():", mask.sum())
            # print("@@@@@@@@@@@@@@@@@ kd_hm_loss_raw:", kd_hm_loss_raw)
        elif loss_cfg.HM_LOSS.type in ['FocalLoss']:
            batch_size = int(cls_pred_tea.shape[0])
            cls_labels = self.dense_head.forward_ret_dict['box_cls_labels']
            # print('@@@@@@@@@ box_cls_labels:', cls_labels.shape)
            batch_cls_preds_stu = cls_pred_stu.view(batch_size, -1, loss_cfg.HM_LOSS['num_class'])
            batch_cls_preds_tea = cls_pred_stu.view(batch_size, -1, loss_cfg.HM_LOSS['num_class'])
            # print('@@@@@@@@@ batch_cls_preds_stu:', batch_cls_preds_stu.shape)
            # print('@@@@@@@@@ batch_cls_preds_tea:', batch_cls_preds_tea.shape)
            if loss_cfg.HM_LOSS.get('fg_mask_anchor', None):
                mid_batch_cls_preds_stu = []
                mid_batch_cls_preds_tea = []
                for i in range(batch_size):
                    cls_label_flag = (cls_labels[i] > 0)
                    mid_cls_preds_stu = batch_cls_preds_stu[i][cls_label_flag]
                    mid_cls_preds_tea = batch_cls_preds_tea[i][cls_label_flag]
                    mid_batch_cls_preds_stu.append(mid_cls_preds_stu)
                    mid_batch_cls_preds_tea.append(mid_cls_preds_tea)
                batch_cls_preds_stu = torch.cat(mid_batch_cls_preds_stu)
                batch_cls_preds_tea = torch.cat(mid_batch_cls_preds_tea)
                # print('@@@@@@@@@ batch_cls_preds_stu:', batch_cls_preds_stu.shape)
                # print('@@@@@@@@@ batch_cls_preds_tea:', batch_cls_preds_tea.shape)

            cls_weights = torch.sum((batch_cls_preds_stu != 10) / loss_cfg.HM_LOSS['num_class'], dim=1).float()
            mid_consis_num = cls_weights.sum(dim=0).float()
            cls_weights /= torch.clamp(mid_consis_num,
                           min=1.0)
            # print('@@@@@@@@@ cls_weights:', cls_weights)
            # print('@@@@@@@@@ cls_weights:', cls_weights.shape)
            # 求Sigmoid Focal Classification Loss
            cls_loss_src = self.kd_hm_loss_func(batch_cls_preds_stu, batch_cls_preds_tea, weights=cls_weights)
            mid_loss_cls = cls_loss_src.sum()
            kd_hm_loss_raw = mid_loss_cls * loss_cfg.HM_LOSS.weight
        else:
            raise NotImplementedError

        kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
        # print("@@@@@@@@@@@@@@@@@ kd_hm_loss:", kd_hm_loss)

        # localization loss
        if loss_cfg.REG_LOSS.weight == 0:
            kd_reg_loss_raw = 0
        elif loss_cfg.REG_LOSS.type == 'WeightedSmoothL1Loss':
            kd_reg_loss_raw = self.cal_kd_reg_loss(box_pred_stu, box_pred_tea)
        else:
            raise NotImplementedError
        kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw
        # print("@@@@@@@@@@@@@@@@@ kd_reg_loss:", kd_reg_loss)

        # TODO: dir loss

        tb_dict['kd_hm_ls'] = kd_hm_loss if isinstance(kd_hm_loss, float) else kd_hm_loss.item()
        tb_dict['kd_loc_ls'] = kd_reg_loss if isinstance(kd_reg_loss, float) else kd_reg_loss.item()

        kd_loss = kd_hm_loss + kd_reg_loss

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

