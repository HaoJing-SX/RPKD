import torch
import torch.nn as nn
import numpy as np

from .anchor_rp_kd_head import AnchorRPKDHead
from .anchor_logit_kd_head import AnchorLogitKDHead
from .anchor_feature_kd_head import AnchorFeatureKDHead
from .anchor_label_kd_head import AnchorLabelAssignKDHead


class AnchorHeadKD(AnchorRPKDHead, AnchorLogitKDHead, AnchorFeatureKDHead, AnchorLabelAssignKDHead):
    def __init__(self, model_cfg, dense_head):
        super(AnchorHeadKD, self).__init__(model_cfg, dense_head)
        self.build_loss(dense_head)

    def get_kd_loss(self, batch_dict, tb_dict):
        kd_loss = 0.0
        # print("@@@@@@@@@@@@@@@@ model_cfg:", self.model_cfg)
        if self.model_cfg.get('RP_KD', None) and self.model_cfg.RP_KD.ENABLED:
            # print("@@@@@@@@@@@@@@@@@ doneRP") # done
            kd_rp_loss, tb_dict = self.get_rp_kd_loss(batch_dict, tb_dict)
            kd_loss += kd_rp_loss

        if self.model_cfg.get('LOGIT_KD', None) and self.model_cfg.LOGIT_KD.ENABLED:
            # print("@@@@@@@@@@@@@@@@@ done1") # done
            kd_logit_loss, tb_dict = self.get_logit_kd_loss(batch_dict, tb_dict)
            kd_loss += kd_logit_loss

        if self.model_cfg.get('FEATURE_KD', None) and self.model_cfg.FEATURE_KD.ENABLED:
            # print("@@@@@@@@@@@@@@@@@ done2") # no
            kd_feature_loss, tb_dict = self.get_feature_kd_loss(
                batch_dict, tb_dict, self.model_cfg.KD_LOSS.FEATURE_LOSS
            )
            kd_loss += kd_feature_loss

        return kd_loss, tb_dict
