import torch
import torch.nn as nn
# import torch_scatter

from functools import partial
from .vfe_template import VFETemplate
from pcdet.utils.spconv_utils import replace_feature, spconv
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.utils import common_utils


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        # print("@@@@@@@@@@@@@@@@@@ voxels_00:", batch_dict['voxels'][0])
        # print("@@@@@@@@@@@@@@@@@@ voxel_features:", batch_dict['voxel_features'])
        # print("@@@@@@@@@@@@@@@@@@ voxel_feature_size:", batch_dict['voxel_features'].shape)

        if self.model_cfg.get('USE_RP', False):
            if self.training:
                voxel_rp = []
                batch_size = batch_dict['batch_size']
                batch_point_index = batch_dict['points'][:, 0]
                batch_voxel_index = batch_dict['voxel_coords'][:, 0]
                point_rp_preds = batch_dict['point_rp_preds'].contiguous()
                # empty_num = 0
                for k in range(batch_size):
                    bs_point_mask = (batch_point_index == k)
                    bs_voxel_mask = (batch_voxel_index == k)
                    now_point_rp_preds = point_rp_preds[bs_point_mask]
                    now_ptv_indexes = batch_dict['ptv_indexes'][bs_point_mask].long()
                    # print('@@@@@@@@@@@@ now_ptv_indexes_size:', now_ptv_indexes.shape)
                    now_voxel_rp = points_mean[:, 3][bs_voxel_mask]
                    now_voxel_rp[now_ptv_indexes] += now_point_rp_preds  # for rppv rppv_nr
                    voxel_rp.append(now_voxel_rp)

                    # now_voxel_rp_mean = torch_scatter.scatter_mean(now_point_rp_preds, now_ptv_indexes, dim=0)
                    # voxel_rp.append(now_voxel_rp_mean)
                    # print('@@@@@@@@@@@@ now_voxel_rp_mean_size:', now_voxel_rp_mean)
                    # print('@@@@@@@@@@@@ now_voxel_rp_mean_size:', now_voxel_rp_mean.shape)
                    # now_num_points = voxel_num_points[bs_voxel_mask]
                    # now_num_bool1 = (now_num_points >= 2)
                    # now_num_bool2 = (now_num_points <= 1)
                    # print('@@@@@@@@@@@@ now_num_points:', now_num_points)
                    # print('@@@@@@@@@@@@ now_num_bool1_sum:', now_num_bool1.sum())
                    # print('@@@@@@@@@@@@ now_num_bool2_sum:', now_num_bool2.sum())

                voxel_mean_rp = torch.cat(voxel_rp, dim=0).unsqueeze(1)
                points_mean[:, 3] = voxel_mean_rp.squeeze(1)
                batch_dict['voxel_features'] = points_mean.contiguous()
            else: # for eval
                voxel_rp = []
                batch_size = batch_dict['batch_size']
                batch_point_index = batch_dict['points'][:, 0]
                batch_voxel_index = batch_dict['voxel_coords'][:, 0]
                point_rp_preds = batch_dict['point_rp_preds'].contiguous()

                voxel_offsets = []
                for k in range(batch_size):
                    bs_voxel_mask = (batch_voxel_index == k)
                    if bs_voxel_mask.any():
                        voxel_offsets.append(bs_voxel_mask.nonzero()[0].min().item())
                    else:
                        voxel_offsets.append(0)
                for k in range(batch_size):
                    bs_point_mask = (batch_point_index == k)
                    bs_voxel_mask = (batch_voxel_index == k)
                    if not bs_point_mask.any() or not bs_voxel_mask.any():
                        if bs_voxel_mask.any():
                            voxel_rp.append(points_mean[:, 3][bs_voxel_mask])
                        continue
                    now_point_rp_preds = point_rp_preds[bs_point_mask]
                    now_ptv_indexes_global = batch_dict['ptv_indexes'][bs_point_mask].long()

                    offset = voxel_offsets[k]
                    now_ptv_indexes_local = now_ptv_indexes_global - offset

                    num_voxels_in_batch = bs_voxel_mask.sum().item()
                    valid_mask = (now_ptv_indexes_local >= 0) & (now_ptv_indexes_local < num_voxels_in_batch)
                    now_ptv_indexes_local = now_ptv_indexes_local[valid_mask]
                    now_point_rp_preds = now_point_rp_preds[valid_mask]
                    now_voxel_rp = points_mean[:, 3][bs_voxel_mask].clone()
                    if len(now_ptv_indexes_local) > 0:
                        now_voxel_rp[now_ptv_indexes_local] += now_point_rp_preds
                    voxel_rp.append(now_voxel_rp)

        # print("@@@@@@@@@@@@ vfe_points:", batch_dict['points'])
        # print("@@@@@@@@@@@@ vfe_voxels:", batch_dict['voxels'])
        # print("@@@@@@@@@@@@ vfe_voxel_features:", batch_dict['voxel_features'])
        # a = {}
        # print("stop:", a[10])
        return batch_dict


class MeanVFERP(VFETemplate):
    def __init__(self, model_cfg, num_point_features, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        # mfe
        self.input_channels = 3 # reserve 3 coordinate columns
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        # print("@@@@@@@@@@@@ model_cfg:", model_cfg)  # {'NAME': 'RPMeanVFE'}
        # print("@@@@@@@@@@@@ num_point_features:", num_point_features)  # 4

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # print("@@@@@@@@@@@@ grid_size:", grid_size)  # [1408 1600 40]
        # print("@@@@@@@@@@@@ sparse_shape:", self.sparse_shape) # [41 1600 1408]

        self.conv_inputrp = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, 16, 3 , padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        self.conv1_rp = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        # self.conv2_rp = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )


        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            # print("@@@@@@@@@@@ src_name:", src_name)
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += cur_num_c_out

        self.num_point_features = c_in
        # print("@@@@@@@@@@@@@@@@@@ num_point_features:", self.num_point_features)

    def get_output_feature_dim(self):
        return self.num_point_features

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # mfe
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        # print("@@@@@@@@@@@@ ret-voxel_features:", voxel_features.shape)
        # print("@@@@@@@@@@@@ ret-voxel_num_points:", voxel_num_points.shape)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # print("@@@@@@@@@@@@ ret-points_mean:", points_mean.shape)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # print("@@@@@@@@@@@@ ret-normalizer:", normalizer.shape)
        points_mean = points_mean / normalizer
        # print("@@@@@@@@@@@@ ret-points_mean:", points_mean.shape)

        voxel_features = points_mean[...,:3].contiguous()
        # print("@@@@@@@@@@@@ voxel_features:", voxel_features)
        # print("@@@@@@@@@@@@ voxel_features_size:", voxel_features.shape)
        # print("@@@@@@@@@@@@ voxel_coords:", batch_dict['voxel_coords'])
        # print("@@@@@@@@@@@@ voxel_coords_size:", batch_dict['voxel_coords'].shape)
        # a = {}
        # print("stop:", a[10])

        batch_size = batch_dict['batch_size']
        voxel_coords = batch_dict['voxel_coords']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # print("@@@@@@@@@@@@ sparse_shape:", self.sparse_shape)

        # kitti [b, 16, 41, 1600, 1408] <- [b, 3, 41, 1600, 1408]
        # waymo [b, 16, 41, 1504, 1504] <- [b, 3, 41, 1504, 1504]
        x = self.conv_inputrp(input_sp_tensor)
        x_conv1 = self.conv1_rp(x)
        x_conv = [x_conv1]
        # # [b, 32, 21, 800, 704] -> [b, 16, 41, 1600, 1408]
        # x_conv2 = self.conv2_rp(x_conv1)
        # x_conv = [x_conv1, x_conv2]

        points = batch_dict["points"]
        new_xyz = points[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (points[:, 0] == k).sum()

        point_features_list = []
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = x_conv[k].indices
            cur_features = x_conv[k].features.contiguous()
            # cur_coords = x_conv1.indices
            # cur_features = x_conv1.features.contiguous()
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)
            # print("@@@@@@@@@@@@ points:", points.shape) # [n, 5]
            # print("@@@@@@@@@@@@ pooled_features:", pooled_features.shape) # [n, 32] [n, 64]

        point_features = torch.cat(point_features_list, dim=-1)
        batch_dict['point_features_rp'] = point_features
        # print("@@@@@@@@@@@@ point_features:", point_features.shape) # [n, 96]

        # print("@@@@@@@@@@@@ before_rp_points:", batch_dict['points'])
        # print("@@@@@@@@@@@@ batch_dict:", batch_dict)
        # a = {}
        # print("stop:", a[10])
        return batch_dict


class MeanVFERPFast(VFETemplate):
    def __init__(self, model_cfg, num_point_features, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        # mfe
        self.input_channels = 3 # reserve 3 coordinate columns
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        # print("@@@@@@@@@@@@ model_cfg:", model_cfg)  # {'NAME': 'RPMeanVFE'}
        # print("@@@@@@@@@@@@ num_point_features:", num_point_features)  # 4

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # print("@@@@@@@@@@@@ grid_size:", grid_size)  # [1408 1600 40]
        # print("@@@@@@@@@@@@ sparse_shape:", self.sparse_shape) # [41 1600 1408]

        num_filters = [16, 16]

        if model_cfg.get('WIDTH', None):
            num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)

        self.conv_inputrp = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, num_filters[0], 3 , padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )

        self.conv1_rp = spconv.SparseSequential(
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        # self.conv2_rp = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )
        #

        c_in = 0
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        for src_name in self.model_cfg.FEATURES_SOURCE:
            # print("@@@@@@@@@@@ src_name:", src_name)
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += cur_num_c_out
        self.num_point_features = c_in
        # print("@@@@@@@@@@@@@@@@@@ num_point_features:", self.num_point_features)

        # self.num_point_features = 32

    def get_output_feature_dim(self):
        return self.num_point_features

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features

    def extract_valid_points_by_batch(self, voxels, voxel_num_points, voxel_coords, batch_size, device=None):
        """

        Args:
            voxels:  [total_voxels, max_points_per_voxel, 4]
            voxel_num_points: [total_voxels]
            voxel_coords: [total_voxels, 4]（

        Returns:
            final_points: [total_valid_points, 5]
                          （
            final_ptv_indexes: [total_valid_points]
        """

        if device is None:
            device = voxels.device
        all_batch_points = []
        all_batch_ptv_indexes = []


        global_batch_voxel_index = voxel_coords[:, 0].long()  #  [total_voxels]
        max_points_per_voxel = voxels.shape[1]
        num_point_features = voxels.shape[2]


        for batch_idx in range(batch_size):
            # step1
            batch_voxel_mask = (global_batch_voxel_index == batch_idx)
            # [batch_voxels, max_points_per_voxel, 4]
            batch_voxels = voxels[batch_voxel_mask]
            # [batch_voxels]
            batch_voxel_num = voxel_num_points[batch_voxel_mask]
            # [batch_voxels]
            batch_global_voxel_idx = torch.nonzero(batch_voxel_mask, as_tuple=True)[0]

            # step2
            point_idx = torch.arange(max_points_per_voxel, device=device).expand(batch_voxels.shape[0], -1)
            valid_mask = point_idx < batch_voxel_num.unsqueeze(1)  # [batch_voxels, max_points_per_voxel]
            # [batch_valid_points, 4]
            batch_voxels_flat = batch_voxels.reshape(-1, num_point_features)  # [batch_voxels*max, 4]
            valid_mask_flat = valid_mask.view(-1)  # [batch_voxels*max]
            batch_points_local = batch_voxels_flat[valid_mask_flat]  # [batch_valid_points, 4]

            batch_voxel_idx_local = torch.arange(batch_voxels.shape[0], device=device).unsqueeze(1).expand(-1,max_points_per_voxel)
            batch_voxel_idx_local_flat = batch_voxel_idx_local.reshape(-1)  # [batch_voxels*max]
            batch_ptv_indexes_local = batch_voxel_idx_local_flat[valid_mask_flat]  # [batch_valid_points]

            # step3：local → global
            batch_ptv_indexes_global = batch_global_voxel_idx[batch_ptv_indexes_local]
            # [batch_valid_points, 5]
            batch_num_col = torch.full(
                size=(batch_points_local.shape[0], 1),
                fill_value=batch_idx,
                dtype=torch.long,
                device=device
            )
            batch_points_with_batch = torch.cat([batch_num_col, batch_points_local], dim=1)

            # step4
            # --------------------------
            all_batch_points.append(batch_points_with_batch)
            all_batch_ptv_indexes.append(batch_ptv_indexes_global)

        if len(all_batch_points) == 0:
            final_points = torch.empty((0, 5), dtype=voxels.dtype, device=device)
            final_ptv_indexes = torch.empty((0,), dtype=torch.long, device=device)
        else:
            final_points = torch.cat(all_batch_points, dim=0)
            final_ptv_indexes = torch.cat(all_batch_ptv_indexes, dim=0)

        return final_points, final_ptv_indexes

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # mfe
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        # print("@@@@@@@@@@@@ ret-voxel_features:", voxel_features.shape)
        # print("@@@@@@@@@@@@ ret-voxel_num_points:", voxel_num_points.shape)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # print("@@@@@@@@@@@@ ret-points_mean:", points_mean.shape)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # print("@@@@@@@@@@@@ ret-normalizer:", normalizer.shape)
        points_mean = points_mean / normalizer
        # print("@@@@@@@@@@@@ ret-points_mean:", points_mean.shape)
        voxel_features = points_mean[...,:3].contiguous()
        # print("@@@@@@@@@@@@ voxel_features:", voxel_features)
        # print("@@@@@@@@@@@@ voxel_features_size:", voxel_features.shape)
        # print("@@@@@@@@@@@@ voxel_coords:", batch_dict['voxel_coords'])
        # print("@@@@@@@@@@@@ voxel_coords_size:", batch_dict['voxel_coords'].shape)
        # a = {}
        # print("stop:", a[10])

        batch_size = batch_dict['batch_size']
        voxel_coords = batch_dict['voxel_coords']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # print("@@@@@@@@@@@@ sparse_shape:", self.sparse_shape)

        # kitti [b, 16, 41, 1600, 1408] <- [b, 3, 41, 1600, 1408]
        # waymo [b, 16, 41, 1504, 1504] <- [b, 3, 41, 1504, 1504]
        x = self.conv_inputrp(input_sp_tensor)
        x_conv1 = self.conv1_rp(x)
        x_conv = [x_conv1]
        # # # [b, 32, 21, 800, 704] -> [b, 16, 41, 1600, 1408]
        # x_conv2 = self.conv2_rp(x_conv1)
        # x_conv = [x_conv1, x_conv2]
        # print("@@@@@@@@@@@@@@@@ points:", batch_dict["points"].shape)

        # for eval
        if not self.training:
            final_points, final_ptv_indexes = self.extract_valid_points_by_batch(
                voxels=batch_dict['voxels'],
                voxel_num_points=batch_dict['voxel_num_points'],
                voxel_coords=batch_dict['voxel_coords'],
                batch_size=batch_dict['batch_size']
            )

            del batch_dict['points']
            del batch_dict['point_num']
            batch_dict['points'] = final_points  # [total_valid_points, 5]
            batch_dict['ptv_indexes'] = final_ptv_indexes  # [total_valid_points]
            batch_dict['point_num'] = final_points.shape[0]
            # print("@@@@@@@@@@@@@@@@ points2:", final_points.shape)

        points = batch_dict["points"]
        new_xyz = points[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (points[:, 0] == k).sum()

        point_features_list = []
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = x_conv[k].indices
            cur_features = x_conv[k].features.contiguous()
            # cur_coords = x_conv1.indices
            # cur_features = x_conv1.features.contiguous()
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)
            # print("@@@@@@@@@@@@ points:", points.shape) # [n, 5]
            # print("@@@@@@@@@@@@ pooled_features:", pooled_features.shape) # [n, 32] [n, 64]

        point_features = torch.cat(point_features_list, dim=-1)
        # point_features = torch.zeros(points.shape[0], self.num_point_features, device="cuda:0")
        batch_dict['point_features_rp'] = point_features
        # print("@@@@@@@@@@@@ point_features:", point_features.shape) # [n, 96]

        # print("@@@@@@@@@@@@ before_rp_points:", batch_dict['points'])
        # print("@@@@@@@@@@@@ batch_dict:", batch_dict)
        # a = {}
        # print("stop:", a[10])
        return batch_dict


