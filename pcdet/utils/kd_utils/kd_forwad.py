import torch
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg
from pcdet.utils import common_utils
from pcdet.models.dense_heads import CenterHead, AnchorHeadTemplate

def adjust_batch_info_teacher(batch):
    # print("@@@@@@@@@@@@ DIFF_VOXEL:", cfg.KD.get('DIFF_VOXEL', None)) # False
    if cfg.KD.get('DIFF_VOXEL', None):
        # print("@@@@@@@@@@@@ done1") # no
        batch['voxels_stu'] = batch.pop('voxels')
        batch['voxel_coords_stu'] = batch.pop('voxel_coords')
        batch['voxel_num_points_stu'] = batch.pop('voxel_num_points')

        batch['voxels'] = batch.pop('voxels_tea')
        batch['voxel_coords'] = batch.pop('voxel_coords_tea')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_tea')

    if cfg.KD.get('SECOND_DATA', False):

        batch['points_stu'] = batch.pop('points')
        batch['voxels_stu'] = batch.pop('voxels')
        batch['voxel_coords_stu'] = batch.pop('voxel_coords')
        batch['voxel_num_points_stu'] = batch.pop('voxel_num_points')
        batch['ptv_indexes_stu'] = batch.pop('ptv_indexes')
        # print("@@@@@@@@@@@@ points:", batch['points_stu'].shape)
        # print("@@@@@@@@@@@@ ptv_indexes:", batch['ptv_indexes'].shape)

        batch['points'] = batch.pop('s_points')
        batch['voxels'] = batch.pop('voxels_tea')
        batch['voxel_coords'] = batch.pop('voxel_coords_tea')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_tea')
        if cfg.KD.get('MMDv3', None):
            batch['ptv_indexes'] = batch.pop('ptv_indexes_tea')
            # print('@@@@@@@@@@@@@@@@@@@@@@ done')
        # print("@@@@@@@@@@@@ s_points:", batch['points'].shape)
        # print("@@@@@@@@@@@@ voxels_add_tea:", batch['voxels'])

    teacher_pred_flag = False
    teacher_target_dict_flag = False
    teacher_decoded_pred_flag = False

    # LOGIT KD
    if cfg.KD.get('LOGIT_KD', None) and cfg.KD.LOGIT_KD.ENABLED:
        if cfg.KD.LOGIT_KD.MODE in ['raw_pred', 'target']:
            # print("@@@@@@@@@@@@ done2") # done
            teacher_pred_flag = True
            teacher_target_dict_flag = True
        elif cfg.KD.LOGIT_KD.MODE == 'decoded_boxes':
            # print("@@@@@@@@@@@@ done3") # no
            teacher_decoded_pred_flag = True
        else:
            raise NotImplementedError

    if cfg.KD.get('LABEL_ASSIGN_KD', None) and cfg.KD.LABEL_ASSIGN_KD.ENABLED:
        # print("@@@@@@@@@@@@ done4") # done
        teacher_decoded_pred_flag = True
    
    if cfg.KD.get('MASK', None):
        if cfg.KD.MASK.get('FG_MASK', None):
            # print("@@@@@@@@@@@@ done5") # no
            teacher_target_dict_flag = True
        
        if cfg.KD.MASK.get('BOX_MASK', None):
            # print("@@@@@@@@@@@@ done6") # no
            teacher_decoded_pred_flag = True
        
        if cfg.KD.MASK.get('SCORE_MASK', None):
            # print("@@@@@@@@@@@@ done7") # no
            teacher_pred_flag = True

    batch['teacher_pred_flag'] = teacher_pred_flag
    batch['teacher_target_dict_flag'] = teacher_target_dict_flag
    batch['teacher_decoded_pred_flag'] = teacher_decoded_pred_flag

# for mmd
def adjust_batch_info_teacher2(batch):
    if cfg.KD.get('SECOND_DATA', False):

        batch['points'] = batch['s_points2']
        batch['voxels'] = batch.pop('voxels_tea2')
        batch['voxel_coords'] = batch['voxel_coords_tea2'] # for rp kd
        batch['voxel_num_points'] = batch.pop('voxel_num_points_tea2')
        batch['ptv_indexes'] = batch.pop('ptv_indexes_tea2')
        # print("@@@@@@@@@@@@ s_points2:", batch['points'].shape)
        # print("@@@@@@@@@@@@ ptv_indexes_tea2:", batch['ptv_indexes'].shape)
        # print("@@@@@@@@@@@@ voxels_add_tea2:", batch['voxels'])

    teacher_rp_pred_flag = False
    teacher_rp_pred_flag2 = False

    if cfg.get('MODEL_TEACHER2', None) and cfg.MODEL_TEACHER2.get('POINT_HEAD_RP', None):
        if cfg.KD.get("MMDv2", False):
            teacher_rp_pred_flag2 = True
            batch['teacher_rp_pred_flag2'] = teacher_rp_pred_flag2
        else:
            teacher_rp_pred_flag = True
            batch['teacher_rp_pred_flag'] = teacher_rp_pred_flag

def adjust_batch_info_student(batch):
    if cfg.KD.get('DIFF_VOXEL', None):
        # print("@@@@@@@@@@@@ done1") # no
        del batch['voxels']
        del batch['voxel_coords']
        del batch['voxel_num_points']

        batch['voxels'] = batch.pop('voxels_stu')
        batch['voxel_coords'] = batch.pop('voxel_coords_stu')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_stu')

    if cfg.KD.get('SECOND_DATA', False):
        del batch['points']
        del batch['voxels']
        del batch['voxel_coords']
        del batch['voxel_num_points']

        batch['points'] = batch.pop('points_stu')
        batch['voxels'] = batch.pop('voxels_stu')
        batch['voxel_coords'] = batch.pop('voxel_coords_stu')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_stu')
        # print("@@@@@@@@@@@@ c_points:", batch['points'].shape)
        # print("@@@@@@@@@@@@ voxels_add_stu:", batch['voxels'])

        if cfg.KD.get('MMD', False):
            del batch['ptv_indexes']
            batch['ptv_indexes'] = batch.pop('ptv_indexes_stu')
            # print("@@@@@@@@@@@@@@@@@@@@@@@@ done")

def add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=None):
    if cfg.KD.get('FEATURE_KD', None) and cfg.KD.FEATURE_KD.ENABLED:
        # print("@@@@@@@@@@@@ done1")# no
        feature_name = cfg.KD.FEATURE_KD.get('FEATURE_NAME_TEA', cfg.KD.FEATURE_KD.FEATURE_NAME)
        batch[feature_name + '_tea'] = batch[feature_name].detach()

    if cfg.KD.get('PILLAR_KD', None) and cfg.KD.PILLAR_KD.ENABLED:
        # print("@@@@@@@@@@@@ done2")# no
        feature_name_tea = cfg.KD.PILLAR_KD.FEATURE_NAME_TEA
        batch['voxel_features_tea'] = batch.pop(feature_name_tea)

    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        # print("@@@@@@@@@@@@ done3")# no
        batch['point_features_tea'] = batch.pop('point_features')
        batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        if cfg.KD.VFE_KD.get('SAVE_INDS', None):
            batch['unq_inv_pfn_tea'] = batch.pop('unq_inv_pfn')
        if cfg.KD.VFE_KD.get('SAVE_3D_FEAT', None):
            batch['spatial_features_tea'] = batch.pop('spatial_features')

    if cfg.KD.get('ROI_KD', None) and cfg.KD.ROI_KD.ENABLED:
        # print("@@@@@@@@@@@@ done4")# no
        batch['rcnn_cls_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_cls')
        batch['rcnn_reg_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_reg')
        batch['roi_head_target_dict_tea'] = teacher_model.roi_head.forward_ret_dict

    if cfg.KD.get('SAVE_COORD_TEA', None):
        # print("@@@@@@@@@@@@ done5")# no
        batch['voxel_coords_tea'] = batch.pop('voxel_coords')
    
    if batch.get('teacher_target_dict_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            # print("@@@@@@@@@@@@ done1")# no
            batch['target_dicts_tea'] = teacher_model.dense_head.forward_ret_dict['target_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            # print("@@@@@@@@@@@@ done2") # done
            batch['spatial_mask_tea'] = batch['spatial_features'].sum(dim=1) != 0

    if batch.get('teacher_pred_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            # print("@@@@@@@@@@@@ done3")# no
            batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            # print("@@@@@@@@@@@@ done4") # done
            batch['cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['cls_preds']
            batch['box_preds_tea'] = teacher_model.dense_head.forward_ret_dict['box_preds']
            batch['dir_cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['dir_cls_preds']

    if batch.get('teacher_decoded_pred_flag', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            # print("@@@@@@@@@@@@ done5") # done
            batch['decoded_pred_tea'] = pred_dicts
        elif isinstance(teacher_model.dense_head, CenterHead):
            # print("@@@@@@@@@@@@ done6")# no
            batch['decoded_pred_tea'] = teacher_model.dense_head.forward_ret_dict['decoded_pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            # print("@@@@@@@@@@@@ done7")# no
            batch['decoded_pred_tea'] = pred_dicts

def add_teacher2_pred_to_batch(teacher_model, batch, pred_dicts=None):
    if batch.get('teacher_rp_pred_flag', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            # print('@@@@@@@@@@@@@@@@@@@@@@ done1')  # done
            batch['sv_rp_pred_tea'] = batch['voxel_features'][:, 3]
            batch['is_mmd'] = True
            # print("@@@@@@@@@@@@ sv_rp_pred_tea:", batch['sv_rp_pred_tea'].shape)  # done

    if batch.get('teacher_rp_pred_flag2', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            # print('@@@@@@@@@@@@@@@@@@@@@@ done2')
            batch['sp_rp_pred_tea'] = batch['point_rp_preds']
            batch['is_mmd'] = True
            # print("@@@@@@@@@@@@ point_rp_pred_tea:", batch['point_rp_pred_tea'].shape)  # done

def forward(model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu, teacher_model2, **kwargs):
    optimizer.zero_grad()
    if extra_optim is not None:
        extra_optim.zero_grad()

    with torch.no_grad():
        adjust_batch_info_teacher(batch)
        load_data_to_gpu(batch)
        if teacher_model.training:
            batch = teacher_model(batch)
            pred_dicts = None

            add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)
        else:
            pred_dicts, ret_dict = teacher_model(batch)
            # print("@@@@@@@@@@@@ pred_dicts:", pred_dicts)
            # print("@@@@@@@@@@@@ point_size:", batch['points'].shape)
            # print("@@@@@@@@@@@@ voxel_features_tea:", batch['voxel_features'].shape)

            add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)
            if cfg.KD.get('MMD', None) and teacher_model2 is not None:
                adjust_batch_info_teacher2(batch)
                pred_dicts2, ret_dict2 = teacher_model2(batch)
                # print("@@@@@@@@@@@@ pred_dicts2:", pred_dicts2)
                # print("@@@@@@@@@@@@ point_size:", batch['points'].shape)
                # print("@@@@@@@@@@@@ voxel_features_tea2:", batch['voxel_features'].shape)

                add_teacher2_pred_to_batch(teacher_model2, batch, pred_dicts=pred_dicts2)

    adjust_batch_info_student(batch)

    ret_dict, tb_dict, disp_dict = model(batch)
    loss = ret_dict['loss'].mean()

    # a = {}
    # print("stop:", a[10])

    loss.backward()
    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

    optimizer.step()
    if extra_optim is not None:
        extra_optim.step()

    return loss, tb_dict, disp_dict
