from .detector3d_template import Detector3DTemplate
from pcdet.models.kd_heads.center_head.center_kd_head import CenterHeadKD
from pcdet.models.kd_heads.anchor_head.anchor_kd_head import AnchorHeadKD

class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # kd_head definition
        # self.kd_head = CenterHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None
        self.kd_head = AnchorHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None

        # kd_head --> dense_head.kd_head
        self.dense_head.kd_head = self.kd_head

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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

            # print("@@@@@@@@@@@@@@@@@ tb_dict:", tb_dict)
            # a = {}
            # print("stop:", a[10])

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn

        if self.model_cfg.get('POINT_HEAD_RP', None):
            loss_rp, tb_dict = self.point_head_rp.get_loss(tb_dict)
            loss += loss_rp

        return loss, tb_dict, disp_dict