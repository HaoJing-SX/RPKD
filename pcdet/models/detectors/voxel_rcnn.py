from .detector3d_template import Detector3DTemplate
from pcdet.models.kd_heads.anchor_head.anchor_kd_head import AnchorHeadKD

class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # kd_head的初始定义
        # self.kd_head = CenterHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None
        self.kd_head = AnchorHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None

        # 将kd_head赋值给dense_head.kd_head
        self.dense_head.kd_head = self.kd_head

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED:
                kd_loss, tb_dict, disp_dict = self.get_kd_loss(batch_dict, tb_dict, disp_dict)
                loss += kd_loss

            # print("@@@@@@@@@@@@@@@@@ tb_dict:", tb_dict)
            ret_dict = {
                'loss': loss
            }

            # a = {}
            # print("stop:", a[10])

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        if self.model_cfg.get('POINT_HEAD_RP', None):
            loss_rp, tb_dict = self.point_head_rp.get_loss(tb_dict)
            loss += loss_rp
            
        return loss, tb_dict, disp_dict
