from .detector3d_template import Detector3DTemplate
from pcdet.models.kd_heads.center_head.center_kd_head import CenterHeadKD
from pcdet.models.kd_heads.anchor_head.anchor_kd_head import AnchorHeadKD

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        if self.dense_head is None and self.dense_head_aux is not None:
            self.dense_head = self.dense_head_aux

        # kd_head definition
        # self.kd_head = CenterHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None
        self.kd_head = AnchorHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None

        # kd_head --> dense_head.kd_head
        self.dense_head.kd_head = self.kd_head

        # # for mmd
        # if model_cfg.get('RP_KD', None):
        #     self.dense_head.sub_kd_head = self.point_head_rp

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.is_teacher and self.training:
            # print("done2") # no done
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED:
                kd_loss, tb_dict, disp_dict = self.get_kd_loss(batch_dict, tb_dict, disp_dict)
                loss += kd_loss
                # print("@@@@@@@@@@@@@@@@@ kd_loss:", kd_loss)
            # print("@@@@@@@@@@@@@@@@@ tb_dict:", tb_dict)
            # print("@@@@@@@@@@@@@@ main_done2")
            ret_dict = {
                'loss': loss
            }

            # a = {}
            # print("stop:", a[10])

            return ret_dict, tb_dict, disp_dict
        else:
            # print("batch_dict:", batch_dict)
            # print("@@@@@@@@@@@@@@@ main_done1")

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        if self.model_cfg.get('POINT_HEAD_RP', None):
            loss_rp, tb_dict = self.point_head_rp.get_loss(tb_dict)
            loss += loss_rp
        
        return loss, tb_dict, disp_dict
