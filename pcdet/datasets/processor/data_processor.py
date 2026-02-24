from functools import partial

import numpy as np
import math
import copy
from skimage import transform
import cv2

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
            data_dict['point_num'] = data_dict['points'].shape[0]

        if data_dict.get('s_points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['s_points'], self.point_cloud_range)
            data_dict['s_points'] = data_dict['s_points'][mask]
            data_dict['s_point_num'] = data_dict['s_points'].shape[0]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

            if data_dict.get('s_points', None) is not None:
                s_points = data_dict['s_points']
                shuffle_idx = np.random.permutation(s_points.shape[0])
                s_points = s_points[shuffle_idx]
                data_dict['s_points'] = s_points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']

        limit_range = self.point_cloud_range
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['points_copy'] = points_copy

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        # print("@@@@@@@@@@@@@@@@@@@@ points_copy_size:", data_dict['points_copy'].shape)
        # a = {}
        # print("stop:", a[10])
        return data_dict

    # only rpv
    def transform_points_to_voxels_rpv(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_rpv, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range
        # second points
        s_points = data_dict['s_points']
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
        s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
               & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
               & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
        s_points = s_points[s_mask]
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
        s_voxel_output = self.voxel_generator.generate(s_points)
        s_voxels, s_coordinates, s_num_points = s_voxel_output
        if not data_dict['use_lead_xyz']:
            s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # mfe
        s_voxel_mean = np.sum(s_voxels, axis=1)
        normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
        s_voxel_mean = s_voxel_mean / normalizer

        # data_dict['voxels_second'] = s_voxels
        # data_dict['voxel_coords_second'] = s_coordinates
        # data_dict['voxel_num_points_second'] = s_num_points
        # data_dict['voxel_mean_second'] = s_voxel_mean

        # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
        # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
        # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
        # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
        # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
        # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

        x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
        y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
        z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

        total_num = x_num * y_num * z_num
        voxel_hash_index = np.arange(0, total_num, 1)
        coords = np.array(s_coordinates)
        voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
        voxel_index = np.arange(0, len(s_voxels), 1)
        voxel_hash_ori_bool = (voxel_hash_index == -1)
        voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
        voxel_hash_ori_bool[voxel_coords_product] = True
        voxel_hash_ori_index[voxel_coords_product] = voxel_index
        # voxel_hash_ori_reflect[voxel_coords_product] = s_voxel_mean[:, 3]
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_bool:', voxel_hash_ori_bool.sum())  # n
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_index:', voxel_hash_ori_index.max())  # n-1
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_reflect:', voxel_hash_ori_reflect.max())  # 0.99
        # print('@@@@@@@@@@@@@@ limit_range:', limit_range)  # [  0.  -40.   -3.   70.4  40.    1. ]

        # main points
        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        intensity = np.zeros(points.shape[0], dtype=float)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        # print("@@@@@@@@@@@@ intensity:", intensity)
        # print("@@@@@@@@@@@@ points_size:", points.shape) # (14774, 4)
        # print("@@@@@@@@@@@@ intensity_size:", intensity.shape)

        # assign s_voxel_mean intensity to points
        direct_align_num = 0
        near_align_num = 0
        unalign_num = 0
        limit_xcoord = [0, x_num - 1]
        limit_ycoord = [0, y_num - 1]
        limit_zcoord = [0, z_num - 1]
        for i in range(len(points)):
        # for i in range(3):
            now_coords = [0, 0, 0]
            # nz, ny, nx
            now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
            now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
            now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
            now_coords = np.floor(now_coords).astype(int)
            # print("@@@@@@@@@@@@ points:", points[i])
            # print("@@@@@@@@@@@@ now_coords:", now_coords)
            now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
            if voxel_hash_ori_bool[now_product]:
                intensity[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                direct_align_num += 1
                # print("@@@@@@@@@@@@ d_s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]])
                # print("@@@@@@@@@@@@ d_intensity:", i, " ", intensity[i])
            else:
                z_coords = np.clip([now_coords[0] - 2, now_coords[0] - 1, now_coords[0], now_coords[0] + 1, now_coords[0] + 2],
                                   limit_zcoord[0], limit_zcoord[1])
                y_coords = np.clip([now_coords[1] - 2, now_coords[1] - 1, now_coords[1], now_coords[1] + 1, now_coords[1] + 2],
                                   limit_ycoord[0], limit_ycoord[1])
                x_coords = np.clip([now_coords[2] - 2, now_coords[2] - 1, now_coords[2], now_coords[2] + 1, now_coords[2] + 2],
                                   limit_xcoord[0], limit_xcoord[1])
                # print("@@@@@@@@@@@@@@ z_coords:", z_coords)
                # print("@@@@@@@@@@@@@@ y_coords:", y_coords)
                # print("@@@@@@@@@@@@@@ x_coords:", x_coords)
                stereo_voxel_product = []
                stereo_distance = []
                for iz in range(5):
                    now_coords[0] = z_coords[iz]
                    for iy in range(5):
                        now_coords[1] = y_coords[iy]
                        for ix in range(5):
                            now_coords[2] = x_coords[ix]
                            now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                            if voxel_hash_ori_bool[now_product]:
                                now_voxel_center = np.array([0.00, 0.00, 0.00])
                                now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                stereo_voxel_product.append(now_product)
                                stereo_distance.append(now_distance)
                            else:
                                stereo_voxel_product.append(-1)
                                stereo_distance.append(100)

                if np.max(stereo_voxel_product) >= 0:
                    min_distance_idx = np.argmin(np.array(stereo_distance))
                    intensity[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                    near_align_num += 1
                else:
                    intensity[i] = 0
                    unalign_num += 1
                # print("@@@@@@@@@@@@ n_intensity:", i, " ", intensity[i])
        points[:, 3] = intensity

        # cv2.imwrite('image2.png', img)
        # cv2.imshow('bev.png', img)
        # cv2.waitKey(-1)
        # a = {}
        # print("stop:", a[10])

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['points'] = points
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        # print("@@@@@@@@@@@@ data_dict-point_num:", data_dict['point_num'])
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        return data_dict

    # for rppv_nr
    def transform_points_to_voxels_rp_nr(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_rp_nr, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # mfe
        voxel_mean = np.sum(voxels, axis=1)
        normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
        voxel_mean = voxel_mean / normalizer

        x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
        y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
        z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
        total_num = x_num * y_num * z_num
        voxel_hash_index = np.arange(0, total_num, 1)

        coords = np.array(coordinates)
        voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
        voxel_index = np.arange(0, len(voxels), 1)
        voxel_hash_ori_bool = (voxel_hash_index == -1)
        voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
        voxel_hash_ori_bool[voxel_coords_product] = True
        voxel_hash_ori_index[voxel_coords_product] = voxel_index

        ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
        ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
        ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
        ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

        ptv_successnum = 0
        ptv_failnum = 0
        point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0
        ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
        ptv_positive_mask = (ptv_indexes == 1)  # initialize false
        # print("@@@@@@@@@@@@@@@ voxel_mean:", voxel_mean[0])
        # print("@@@@@@@@@@@@@@@ voxel_mean00:", voxel_mean[voxel_hash_ori_index[ptv_product[0]]][3])
        for i in range(len(points)):
            if voxel_hash_ori_bool[ptv_product[i]]:
                point_mfe_rp_labels[i] = voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3]
                ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                ptv_positive_mask[i] = True
                ptv_successnum += 1
            else:
                ptv_failnum += 1

        if ptv_failnum > 0:
            points = points[ptv_positive_mask]
            point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
            ptv_indexes = ptv_indexes[ptv_positive_mask]

        data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
        data_dict['ptv_indexes'] = ptv_indexes

        points[:, 3] = 0
        data_dict['points'] = points
        # print("@@@@@@@@@@@@ points0:", points[0])
        # print("@@@@@@@@@@@@ point_size:", points.shape)

        voxels[:, :, 3] = 0
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        # print("@@@@@@@@@@@@ voxels_size:", voxels.shape)

        # a = {}
        # print("stop:", a[10])

        return data_dict

    # only rppv
    def transform_points_to_voxels_rp(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_rp, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        if self.mode == 'train':
            # second points - ori
            s_points = data_dict['s_points']
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
            s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                   & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                   & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
            s_points = s_points[s_mask]
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
            s_voxel_output = self.voxel_generator.generate(s_points)
            s_voxels, s_coordinates, s_num_points = s_voxel_output
            if not data_dict['use_lead_xyz']:
                # print("@@@@@@@@@@@@@@ done1") # no
                s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
            # mfe ori point - voxels
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer
            # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
            # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
            # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
            # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
            # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
            # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # ori
            coords = np.array(s_coordinates)
            # print('@@@@@@@@@@@@@@ coords:', coords)  # ([nz, ny, nx])
            # print("@@@@@@@@@@@@ coords[0]max:", coords[:, 0].max())
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            point_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            ptsv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            # point_align_mask = (point_rp_labels == 1) # initialize false
            # print("@@@@@@@@@@@@ points_size:", points.shape)

            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]
            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)

                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3]
                    ptsv_indexes[i] = voxel_hash_ori_index[now_product]
                    direct_align_num += 1
                    # point_align_mask[i] = True
                    # print("@@@@@@@@@@@@ d_s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]])
                    # print("@@@@@@@@@@@@ d_intensity:", i, " ", intensity[i])
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel_center = np.array([0.00, 0.00, 0.00])
                                    now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                    now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                    now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                    # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                    # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                    # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                    now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                    # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                    # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                    stereo_voxel_product.append(now_product)
                                    stereo_distance.append(now_distance)
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3]
                        ptsv_indexes[i] = voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]
                        near_align_num += 1
                        # point_align_mask[i] = True
                    else:
                        point_rp_labels[i] = 0
                        ptsv_indexes[i] = -1
                        unalign_num += 1
                    # print("@@@@@@@@@@@@ n_intensity:", i, " ", intensity[i])

            points[:, 3] = point_rp_labels
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            # mfe
            voxel_mean = np.sum(voxels, axis=1)
            normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            voxel_mean = voxel_mean / normalizer
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index
            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    point_mfe_rp_labels[i] = voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3]
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                # point_rp_labels = point_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsv_indexes = ptsv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsv_indexes'] = ptsv_indexes
        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points
            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])
        return data_dict

    # only rppv_fast
    def transform_points_to_voxels_rpfast(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_rpfast, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)
        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)
        # print('@@@@@@@@@@@@@@@@@ voxels:', voxels)
        # print('@@@@@@@@@@@@@@@@@ voxel_num_points:', num_points)
        # print('@@@@@@@@@@@@@@@@@ voxel_num_points:', num_points)
        # a = {}
        # print("stop:", a[10])

        if self.mode == 'train':
            # second points - ori
            s_points = data_dict['s_points']
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
            s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                   & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                   & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
            s_points = s_points[s_mask]
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
            s_voxel_output = self.voxel_generator.generate(s_points)
            s_voxels, s_coordinates, s_num_points = s_voxel_output
            if not data_dict['use_lead_xyz']:
                # print("@@@@@@@@@@@@@@ done1") # no
                s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
            # mfe ori point - voxels
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer
            # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
            # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
            # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
            # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
            # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
            # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # ori
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            # assign s_voxel_mean intensity to points
            point_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            ptsv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            # point_align_mask = (point_rp_labels == 1) # initialize false
            # print("@@@@@@@@@@@@ points_size:", points.shape)

            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]
            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)

                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3]
                    ptsv_indexes[i] = voxel_hash_ori_index[now_product]
                    direct_align_num += 1
                    # point_align_mask[i] = True
                    # print("@@@@@@@@@@@@ d_s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]])
                    # print("@@@@@@@@@@@@ d_intensity:", i, " ", intensity[i])
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel_center = np.array([0.00, 0.00, 0.00])
                                    now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                    now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                    now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                    # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                    # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                    # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                    now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                    # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                    # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                    stereo_voxel_product.append(now_product)
                                    stereo_distance.append(now_distance)
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3]
                        ptsv_indexes[i] = voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]
                        near_align_num += 1
                        # point_align_mask[i] = True
                    else:
                        point_rp_labels[i] = 0
                        ptsv_indexes[i] = -1
                        unalign_num += 1
                    # print("@@@@@@@@@@@@ n_intensity:", i, " ", intensity[i])

            points[:, 3] = point_rp_labels
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            # mfe
            voxel_mean = np.sum(voxels, axis=1)
            normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            voxel_mean = voxel_mean / normalizer

            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index
            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0

            ptv_indexes = voxel_hash_ori_index[ptv_product]
            ptv_positive_mask = (ptv_indexes != -1)
            ptv_successnum = ptv_positive_mask.sum()
            ptv_failnum = len(points) - ptv_successnum

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                # point_rp_labels = point_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsv_indexes = ptsv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsv_indexes'] = ptsv_indexes
        else:
            pass

        # a = {}
        # print("stop:", a[10])
        return data_dict

    # for rppv_v2
    def transform_points_to_voxels_rp_v2(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_rp_v2, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        if self.mode == 'train':
            # second points - ori
            s_points = data_dict['s_points']
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
            s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                   & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                   & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
            s_points = s_points[s_mask]
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
            s_voxel_output = self.voxel_generator.generate(s_points)
            s_voxels, s_coordinates, s_num_points = s_voxel_output
            if not data_dict['use_lead_xyz']:
                # print("@@@@@@@@@@@@@@ done1") # no
                s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
            # mfe ori point - voxels
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer
            # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
            # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
            # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
            # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
            # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
            # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # ori
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            # assign s_voxel_mean intensity to points
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            point_align_mask = (point_mfe_rp_labels == 1) # initialize false
            nearest_spoints = []
            nearest_svoxel_coords = []

            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]
            # point match
            for i in range(len(points)):
            # for i in range(5):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]

                now_coords = np.floor(now_coords).astype(int)
                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                    direct_align_num += 1
                    point_align_mask[i] = True
                    stereo_distance = []
                    stereo_points = []
                    now_voxel = s_voxels[voxel_hash_ori_index[now_product]]
                    now_point_num = s_num_points[voxel_hash_ori_index[now_product]]
                    for ip in range(now_point_num):
                        stereo_distance.append(np.sqrt(sum(np.power((points[i][0:3] - now_voxel[ip][0:3]), 2))))
                        stereo_points.append(now_voxel[ip])
                    min_distance_idx = np.argmin(np.array(stereo_distance))
                    nearest_spoints.append(stereo_points[min_distance_idx])
                    nearest_svoxel_coords.append(s_coordinates[voxel_hash_ori_index[now_product]])
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    stereo_points = []
                    stereo_voxel_coords = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel = s_voxels[voxel_hash_ori_index[now_product]]
                                    now_point_num = s_num_points[voxel_hash_ori_index[now_product]]
                                    for ip in range(now_point_num):
                                        stereo_voxel_product.append(now_product)
                                        stereo_distance.append(np.sqrt(sum(np.power((points[i][0:3] - now_voxel[ip][0:3]), 2))))
                                        stereo_points.append(now_voxel[ip])
                                        stereo_voxel_coords.append(s_coordinates[voxel_hash_ori_index[now_product]])
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                                    stereo_points.append([0, 0, 0, 0])
                                    stereo_voxel_coords.append([-1, -1, -1])
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                        nearest_spoints.append(stereo_points[min_distance_idx])
                        nearest_svoxel_coords.append(stereo_voxel_coords[min_distance_idx])
                        near_align_num += 1
                        point_align_mask[i] = True
                    else:
                        point_mfe_rp_labels[i] = 0
                        nearest_spoints.append([0, 0, 0, 0])
                        nearest_svoxel_coords.append([-1, -1, -1]) # mark empty
                        unalign_num += 1
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_voxel_product:", stereo_voxel_product)
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_distance:", stereo_distance)
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_points:", stereo_points)
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_voxel_coords:", stereo_voxel_coords)
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_svoxel_coords:", stereo_svoxel_coords)
                # print("@@@@@@@@@@@@@@@@@@@@ stereo_set_coords:", stereo_set_coords)

            # for ptsp_indexes
            ptsp_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            mid_indexes = np.arange(0, s_points.shape[0], 1)
            for i in range(len(points)):
            # for i in range(5):
                if np.min(nearest_svoxel_coords[i]) >= 0:
                    now_svoxel_coords = nearest_svoxel_coords[i]
                    limit_px = [now_svoxel_coords[2] * (config.VOXEL_SIZE[0]) + limit_range[0],
                                (now_svoxel_coords[2] + 1) * (config.VOXEL_SIZE[0]) + limit_range[0]]
                    limit_py = [now_svoxel_coords[1] * (config.VOXEL_SIZE[1]) + limit_range[1],
                                (now_svoxel_coords[1] + 1) * (config.VOXEL_SIZE[1]) + limit_range[1]]
                    limit_pz = [now_svoxel_coords[0] * (config.VOXEL_SIZE[2]) + limit_range[2],
                                (now_svoxel_coords[0] + 1) * (config.VOXEL_SIZE[2]) + limit_range[2]]
                    now_spoint_indexes = mid_indexes[(s_points[:, 0] >= limit_px[0]) & (s_points[:, 0] <= limit_px[1]) &
                                                     (s_points[:, 1] >= limit_py[0]) & (s_points[:, 1] <= limit_py[1]) &
                                                     (s_points[:, 2] >= limit_pz[0]) & (s_points[:, 2] <= limit_pz[1])]
                    now_distance = []
                    for j in range(len(now_spoint_indexes)):
                        mid_sp_index = now_spoint_indexes[j]
                        now_distance.append(np.sqrt(sum(np.power((points[i][0:3] - s_points[mid_sp_index][0:3]), 2))))
                    # for few boundary points
                    if len(now_distance) == 0:
                        ptsp_indexes[i] = -1
                    else:
                        min_distance_idx = np.argmin(np.array(now_distance))
                        ptsp_indexes[i] = now_spoint_indexes[min_distance_idx]
                    # print(i, " @@@@@@@@@@@@@@@@@@@@ now_spoint_indexes:", now_spoint_indexes, "ptsp_indexes:", ptsp_indexes[i])
                else:
                    ptsp_indexes[i] = -1 # ptsp for empty
                    # print(i, " @@@@@@@@@@@@@@@@@@@@ ptsp_indexes:", ptsp_indexes[i])
            # print("@@@@@@@@@@@@ ptsp_indexes_size:", ptsp_indexes.shape)

            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsp_indexes = ptsp_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsp_indexes'] = ptsp_indexes  # for future ptsp_kd
            # print("@@@@@@@@@@@@ ptsp_indexes_size:", ptsp_indexes.shape)

            data_dict.pop('s_points', None)
            data_dict.pop('s_point_num', None)
        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points
            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])
        return data_dict

    # for rppv_mmdv2
    def transform_points_to_voxels_compress_mmdv2(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_mmdv2, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points # as student
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        # second points - ori
        s_points = data_dict['s_points']
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
        s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                 & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                 & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
        s_points = s_points[s_mask]
        s_points_copy = copy.deepcopy(s_points)
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
        data_dict['s_points'] = s_points_copy
        data_dict['s_point_num'] = s_points.shape[0]
        s_voxel_output = self.voxel_generator.generate(s_points)
        s_voxels, s_coordinates, s_num_points = s_voxel_output
        s_voxels_copy = copy.deepcopy(s_voxels)
        if not data_dict['use_lead_xyz']:
            s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels_tea'] = s_voxels_copy
        data_dict['voxel_coords_tea'] = s_coordinates
        data_dict['voxel_num_points_tea'] = s_num_points # as teacher

        if self.mode == 'train':
            # mfe second points - ori
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            # print('@@@@@@@@@@@@@@ voxel_hash_index:', voxel_hash_index) # [   0    1    2 ... 90111997 90111998 90111999]

            # ori point - voxels
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            # for rp_nr
            ptv_coordz = np.floor((s_points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((s_points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((s_points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            spoint_mfe_rp_labels = np.zeros(s_points.shape[0], dtype=float)  # initialize 0

            sptsv_indexes = np.zeros(s_points.shape[0], dtype=int)  # initialize 0
            sptsv_positive_mask = (sptsv_indexes == 1)  # initialize false
            for i in range(len(s_points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    spoint_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3:]
                    sptsv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    sptsv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                s_points = s_points[sptsv_positive_mask]
                spoint_mfe_rp_labels = spoint_mfe_rp_labels[sptsv_positive_mask]
                sptsv_indexes = sptsv_indexes[sptsv_positive_mask]

            data_dict['ptv_indexes_tea2'] = sptsv_indexes

            s_points[:, 3] = 0
            data_dict['s_points2'] = s_points

            s_voxels[:, :, 3] = 0
            data_dict['voxels_tea2'] = s_voxels
            data_dict['voxel_coords_tea2'] = s_coordinates
            data_dict['voxel_num_points_tea2'] = s_num_points # as teacher

            # main points - compress
            # assign s_voxel_mean intensity to points
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            point_align_mask = (point_mfe_rp_labels == 1) # initialize false
            nearest_spoints = []
            nearest_svoxel_coords = []

            # assign s_voxel_mean intensity to points
            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]

            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)
                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                    direct_align_num += 1
                    point_align_mask[i] = True
                    stereo_distance = []
                    stereo_points = []
                    now_voxel = s_voxels[voxel_hash_ori_index[now_product]]
                    now_point_num = s_num_points[voxel_hash_ori_index[now_product]]
                    for ip in range(now_point_num):
                        stereo_distance.append(np.sqrt(sum(np.power((points[i][0:3] - now_voxel[ip][0:3]), 2))))
                        stereo_points.append(now_voxel[ip])
                    min_distance_idx = np.argmin(np.array(stereo_distance))
                    nearest_spoints.append(stereo_points[min_distance_idx])
                    nearest_svoxel_coords.append(s_coordinates[voxel_hash_ori_index[now_product]])
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    stereo_points = []
                    stereo_voxel_coords = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel = s_voxels[voxel_hash_ori_index[now_product]]
                                    now_point_num = s_num_points[voxel_hash_ori_index[now_product]]
                                    for ip in range(now_point_num):
                                        stereo_voxel_product.append(now_product)
                                        stereo_distance.append(np.sqrt(sum(np.power((points[i][0:3] - now_voxel[ip][0:3]), 2))))
                                        stereo_points.append(now_voxel[ip])
                                        stereo_voxel_coords.append(s_coordinates[voxel_hash_ori_index[now_product]])
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                                    stereo_points.append([0, 0, 0, 0])
                                    stereo_voxel_coords.append([-1, -1, -1])

                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                        nearest_spoints.append(stereo_points[min_distance_idx])
                        nearest_svoxel_coords.append(stereo_voxel_coords[min_distance_idx])
                        near_align_num += 1
                        point_align_mask[i] = True
                    else:
                        point_mfe_rp_labels[i] = 0
                        nearest_spoints.append([0, 0, 0, 0])
                        nearest_svoxel_coords.append([-1, -1, -1]) # mark empty
                        unalign_num += 1

            # for ptsp_indexes
            ptsp_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            mid_indexes = np.arange(0, s_points.shape[0], 1)
            for i in range(len(points)):
            # for i in range(5):
                if np.min(nearest_svoxel_coords[i]) >= 0:
                    now_svoxel_coords = nearest_svoxel_coords[i]
                    limit_px = [now_svoxel_coords[2] * (config.VOXEL_SIZE[0]) + limit_range[0],
                                (now_svoxel_coords[2] + 1) * (config.VOXEL_SIZE[0]) + limit_range[0]]
                    limit_py = [now_svoxel_coords[1] * (config.VOXEL_SIZE[1]) + limit_range[1],
                                (now_svoxel_coords[1] + 1) * (config.VOXEL_SIZE[1]) + limit_range[1]]
                    limit_pz = [now_svoxel_coords[0] * (config.VOXEL_SIZE[2]) + limit_range[2],
                                (now_svoxel_coords[0] + 1) * (config.VOXEL_SIZE[2]) + limit_range[2]]
                    now_spoint_indexes = mid_indexes[(s_points[:, 0] >= limit_px[0]) & (s_points[:, 0] <= limit_px[1]) &
                                                     (s_points[:, 1] >= limit_py[0]) & (s_points[:, 1] <= limit_py[1]) &
                                                     (s_points[:, 2] >= limit_pz[0]) & (s_points[:, 2] <= limit_pz[1])]
                    now_distance = []
                    for j in range(len(now_spoint_indexes)):
                        mid_sp_index = now_spoint_indexes[j]
                        now_distance.append(np.sqrt(sum(np.power((points[i][0:3] - s_points[mid_sp_index][0:3]), 2))))
                    # for few boundary points
                    if len(now_distance) == 0:
                        ptsp_indexes[i] = -1
                    else:
                        min_distance_idx = np.argmin(np.array(now_distance))
                        ptsp_indexes[i] = now_spoint_indexes[min_distance_idx]
                    # print(i, " @@@@@@@@@@@@@@@@@@@@ now_spoint_indexes:", now_spoint_indexes, "ptsp_indexes:", ptsp_indexes[i])
                else:
                    ptsp_indexes[i] = -1 # ptsp for empty
                    # print(i, " @@@@@@@@@@@@@@@@@@@@ ptsp_indexes:", ptsp_indexes[i])
            # print("@@@@@@@@@@@@ ptsp_indexes_size:", ptsp_indexes.shape)

            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsp_indexes = ptsp_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsp_indexes'] = ptsp_indexes  # for future ptsp_kd

        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # compress
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index
            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points

            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])

        return data_dict

    # only kd
    def transform_points_to_voxels_compress_kd(self, data_dict=None, config=None):
        # print("@@@@@@@ done1")
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_kd, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@ done1") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # else:
        #     print("@@@@@@@@@@ done2")
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        points = data_dict['s_points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@ done1") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # else:
        #     print("@@@@@@@@@@ done2")
        data_dict['voxels_tea'] = voxels
        data_dict['voxel_coords_tea'] = coordinates
        data_dict['voxel_num_points_tea'] = num_points

        # print("@@@@@@@@@@@@ points:", data_dict['points'])
        # print("@@@@@@@@@@@@ s_points:", data_dict['s_points'])
        # print("@@@@@@@@@@@@ voxels:", data_dict['voxels'])
        # print("@@@@@@@@@@@@ voxels_tea:", data_dict['voxels_tea'])
        #
        # a = {}
        # print("stop:", a[10])

        return data_dict

    # rpv and kd
    def transform_points_to_voxels_compress_rkd(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_rkd, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range
        # second points
        s_points = data_dict['s_points']
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
        s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
               & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
               & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
        s_points = s_points[s_mask]
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
        s_voxel_output = self.voxel_generator.generate(s_points)
        s_voxels, s_coordinates, s_num_points = s_voxel_output
        if not data_dict['use_lead_xyz']:
            s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # mfe
        s_voxel_mean = np.sum(s_voxels, axis=1)
        normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
        s_voxel_mean = s_voxel_mean / normalizer

        # data_dict['voxels_second'] = s_voxels
        # data_dict['voxel_coords_second'] = s_coordinates
        # data_dict['voxel_num_points_second'] = s_num_points
        # data_dict['voxel_mean_second'] = s_voxel_mean

        # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
        # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
        # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
        # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
        # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
        # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

        x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
        y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
        z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])

        total_num = x_num * y_num * z_num
        voxel_hash_index = np.arange(0, total_num, 1)
        coords = np.array(s_coordinates)
        voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
        voxel_index = np.arange(0, len(s_voxels), 1)
        voxel_hash_ori_bool = (voxel_hash_index == -1)
        voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
        voxel_hash_ori_bool[voxel_coords_product] = True
        voxel_hash_ori_index[voxel_coords_product] = voxel_index
        # voxel_hash_ori_reflect[voxel_coords_product] = s_voxel_mean[:, 3]
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_bool:', voxel_hash_ori_bool.sum())  # n
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_index:', voxel_hash_ori_index.max())  # n-1
        # print('@@@@@@@@@@@@@@ voxel_hash_ori_reflect:', voxel_hash_ori_reflect.max())  # 0.99
        # print('@@@@@@@@@@@@@@ limit_range:', limit_range)  # [  0.  -40.   -3.   70.4  40.    1. ]

        # main points
        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        intensity = np.zeros(points.shape[0], dtype=float)
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ intensity:", intensity)
        # print("@@@@@@@@@@@@ points_size:", points.shape) # (14774, 4)
        # print("@@@@@@@@@@@@ intensity_size:", intensity.shape)

        # assign s_voxel_mean intensity to points
        direct_align_num = 0
        near_align_num = 0
        unalign_num = 0
        limit_xcoord = [0, x_num - 1]
        limit_ycoord = [0, y_num - 1]
        limit_zcoord = [0, z_num - 1]
        for i in range(len(points)):
        # for i in range(3):
            now_coords = [0, 0, 0]
            # nz, ny, nx
            now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
            now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
            now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
            now_coords = np.floor(now_coords).astype(int)
            now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
            if voxel_hash_ori_bool[now_product]:
                intensity[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                direct_align_num += 1
                # print("@@@@@@@@@@@@ d_s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]])
                # print("@@@@@@@@@@@@ d_intensity:", i, " ", intensity[i])
            else:
                z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                # z_coords = np.clip([now_coords[0] - 2, now_coords[0] - 1, now_coords[0], now_coords[0] + 1, now_coords[0] + 2],
                #                    limit_zcoord[0], limit_zcoord[1])
                # y_coords = np.clip([now_coords[1] - 2, now_coords[1] - 1, now_coords[1], now_coords[1] + 1, now_coords[1] + 2],
                #                    limit_ycoord[0], limit_ycoord[1])
                # x_coords = np.clip([now_coords[2] - 2, now_coords[2] - 1, now_coords[2], now_coords[2] + 1, now_coords[2] + 2],
                #                    limit_xcoord[0], limit_xcoord[1])
                # print("@@@@@@@@@@@@@@ z_coords:", z_coords)
                # print("@@@@@@@@@@@@@@ y_coords:", y_coords)
                # print("@@@@@@@@@@@@@@ x_coords:", x_coords)
                stereo_voxel_product = []
                stereo_distance = []
                for iz in range(3):
                    now_coords[0] = z_coords[iz]
                    for iy in range(3):
                        now_coords[1] = y_coords[iy]
                        for ix in range(3):
                            now_coords[2] = x_coords[ix]
                            now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                            if voxel_hash_ori_bool[now_product]:
                                now_voxel_center = np.array([0.00, 0.00, 0.00])
                                now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                stereo_voxel_product.append(now_product)
                                stereo_distance.append(now_distance)
                            else:
                                stereo_voxel_product.append(-1)
                                stereo_distance.append(100)

                if np.max(stereo_voxel_product) >= 0:
                    min_distance_idx = np.argmin(np.array(stereo_distance))
                    intensity[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                    near_align_num += 1
                else:
                    intensity[i] = 0
                    unalign_num += 1

        points[:, 3] = intensity

        # a = {}
        # print("stop:", a[10])

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['points'] = points
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        # print("@@@@@@@@@@@@ data_dict-point_num:", data_dict['point_num'])
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        points = data_dict['s_points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@ done1") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # else:
        #     print("@@@@@@@@@@ done2")
        data_dict['voxels_tea'] = voxels
        data_dict['voxel_coords_tea'] = coordinates
        data_dict['voxel_num_points_tea'] = num_points

        # print("@@@@@@@@@@@@ points:", data_dict['points'])
        # print("@@@@@@@@@@@@ s_points:", data_dict['s_points'])
        # print("@@@@@@@@@@@@ voxels:", data_dict['voxels'])
        # print("@@@@@@@@@@@@ voxels_tea:", data_dict['voxels_tea'])
        #
        # a = {}
        # print("stop:", a[10])

        return data_dict

    # rppv and kd
    def transform_points_to_voxels_compress_rpkd(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_rpkd, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points # as student
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        # second points - ori
        s_points = data_dict['s_points']
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
        s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                 & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                 & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
        s_points = s_points[s_mask]
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
        data_dict['s_points'] = s_points
        data_dict['s_point_num'] = s_points.shape[0]
        s_voxel_output = self.voxel_generator.generate(s_points)
        s_voxels, s_coordinates, s_num_points = s_voxel_output
        if not data_dict['use_lead_xyz']:
            s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels_tea'] = s_voxels
        data_dict['voxel_coords_tea'] = s_coordinates
        data_dict['voxel_num_points_tea'] = s_num_points # as teacher

        if self.mode == 'train':
            # mfe second points - ori
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer

            # data_dict['voxels_second'] = s_voxels
            # data_dict['voxel_coords_second'] = s_coordinates
            # data_dict['voxel_num_points_second'] = s_num_points
            # data_dict['voxel_mean_second'] = s_voxel_mean

            # print("@@@@@@@@@@@@ data_dict-second_point_num:", data_dict['second_point_num'])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'][0:5])
            # print("@@@@@@@@@@@@ data_dict-s_voxel_mean:", s_voxel_mean[0:5])
            # print("@@@@@@@@@@@@ data_dict-normalizer:", normalizer[0:5])
            # print("@@@@@@@@@@@@ data_dict-voxels_second:", data_dict['voxels_second'].shape) # (n, 5, 4)
            # print("@@@@@@@@@@@@ data_dict-voxel_num_points_second:", data_dict['voxel_num_points_second'].shape) # (n,)
            # print("@@@@@@@@@@@@ data_dict-voxel_coords_second:", data_dict['voxel_coords_second'].shape) # (n, 3)
            # print("@@@@@@@@@@@@ data_dict-voxel_mean_second:", data_dict['voxel_mean_second'].shape)  # (n, 4)

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            # print('@@@@@@@@@@@@@@ voxel_hash_index:', voxel_hash_index) # [   0    1    2 ... 90111997 90111998 90111999]

            # ori point - voxels
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index
            # print('@@@@@@@@@@@@@@ voxel_hash_ori_bool:', voxel_hash_ori_bool.sum())  # n
            # print('@@@@@@@@@@@@@@ voxel_hash_ori_index:', voxel_hash_ori_index.max())  # n-1
            # print('@@@@@@@@@@@@@@ limit_range:', limit_range)  # [  0.  -40.   -3.   70.4  40.    1. ]

            # main points - compress
            # assign s_voxel_mean intensity to points
            point_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            point_align_mask = (point_rp_labels == 1) # initialize false
            # print("@@@@@@@@@@@@ points:", points)
            # print("@@@@@@@@@@@@ intensity:", intensity)
            # print("@@@@@@@@@@@@ points_size:", points.shape) # (14774, 4)
            # print("@@@@@@@@@@@@ intensity_size:", intensity.shape)

            # assign s_voxel_mean intensity to points
            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]
            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)
                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                    direct_align_num += 1
                    point_align_mask[i] = True
                    # print("@@@@@@@@@@@@ d_s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]])
                    # print("@@@@@@@@@@@@ d_intensity:", i, " ", intensity[i])
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    # z_coords = np.clip([now_coords[0] - 2, now_coords[0] - 1, now_coords[0], now_coords[0] + 1, now_coords[0] + 2],
                    #                    limit_zcoord[0], limit_zcoord[1])
                    # y_coords = np.clip([now_coords[1] - 2, now_coords[1] - 1, now_coords[1], now_coords[1] + 1, now_coords[1] + 2],
                    #                    limit_ycoord[0], limit_ycoord[1])
                    # x_coords = np.clip([now_coords[2] - 2, now_coords[2] - 1, now_coords[2], now_coords[2] + 1, now_coords[2] + 2],
                    #                    limit_xcoord[0], limit_xcoord[1])
                    # print("@@@@@@@@@@@@@@ z_coords:", z_coords)
                    # print("@@@@@@@@@@@@@@ y_coords:", y_coords)
                    # print("@@@@@@@@@@@@@@ x_coords:", x_coords)
                    stereo_voxel_product = []
                    stereo_distance = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel_center = np.array([0.00, 0.00, 0.00])
                                    now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                    now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                    now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                    # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                    # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                    # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                    now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                    # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                    # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                    stereo_voxel_product.append(now_product)
                                    stereo_distance.append(now_distance)
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                        near_align_num += 1
                        point_align_mask[i] = True
                    else:
                        point_rp_labels[i] = 0
                        unalign_num += 1

            points[:, 3] = point_rp_labels
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            # mfe
            voxel_mean = np.sum(voxels, axis=1)
            normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            voxel_mean = voxel_mean / normalizer
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    point_mfe_rp_labels[i] = voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3:]
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1
            # print("@@@@@@@@@@@@ ptv_successnum:", ptv_successnum, " ptv_failnum:", ptv_failnum)

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                # point_rp_labels = point_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # compress
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num
            ptv_successnum = 0
            ptv_failnum = 0
            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points
            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])

        return data_dict

    # for mmd
    def transform_points_to_voxels_compress_mmd(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_mmd, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points # as student
        # print("@@@@@@@@@@@@ data_dict-voxels:", data_dict['voxels'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_num_points:", data_dict['voxel_num_points'].shape)
        # print("@@@@@@@@@@@@ data_dict-voxel_coords:", data_dict['voxel_coords'].shape)

        # second points - ori
        s_points = data_dict['s_points']
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
        s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                 & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                 & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
        s_points = s_points[s_mask]
        s_points_copy = copy.deepcopy(s_points)
        # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
        data_dict['s_points'] = s_points_copy
        data_dict['s_point_num'] = s_points.shape[0]
        s_voxel_output = self.voxel_generator.generate(s_points)
        s_voxels, s_coordinates, s_num_points = s_voxel_output
        s_voxels_copy = copy.deepcopy(s_voxels)
        if not data_dict['use_lead_xyz']:
            s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels_tea'] = s_voxels_copy
        data_dict['voxel_coords_tea'] = s_coordinates
        data_dict['voxel_num_points_tea'] = s_num_points # as teacher

        if self.mode == 'train':
            # mfe second points - ori
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            # print('@@@@@@@@@@@@@@ voxel_hash_index:', voxel_hash_index) # [   0    1    2 ... 90111997 90111998 90111999]

            # ori point - voxels
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            # for rp_nr
            ptv_coordz = np.floor((s_points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((s_points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((s_points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            spoint_mfe_rp_labels = np.zeros(s_points.shape[0], dtype=float)  # initialize 0

            sptsv_indexes = np.zeros(s_points.shape[0], dtype=int)  # initialize 0
            sptsv_positive_mask = (sptsv_indexes == 1)  # initialize false
            for i in range(len(s_points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    spoint_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3:]
                    sptsv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    sptsv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                s_points = s_points[sptsv_positive_mask]
                spoint_mfe_rp_labels = spoint_mfe_rp_labels[sptsv_positive_mask]
                sptsv_indexes = sptsv_indexes[sptsv_positive_mask]

            data_dict['ptv_indexes_tea2'] = sptsv_indexes

            s_points[:, 3] = 0
            data_dict['s_points2'] = s_points

            s_voxels[:, :, 3] = 0
            data_dict['voxels_tea2'] = s_voxels
            data_dict['voxel_coords_tea2'] = s_coordinates
            data_dict['voxel_num_points_tea2'] = s_num_points # as teacher

            # main points - compress
            # assign s_voxel_mean intensity to points
            point_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0

            ptsv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            point_align_mask = (point_rp_labels == 1) # initialize false
            # assign s_voxel_mean intensity to points
            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]

            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)
                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3:]
                    ptsv_indexes[i] = voxel_hash_ori_index[now_product]
                    direct_align_num += 1
                    point_align_mask[i] = True
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel_center = np.array([0.00, 0.00, 0.00])
                                    now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                    now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                    now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                    # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                    # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                    # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                    now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                    # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                    # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                    stereo_voxel_product.append(now_product)
                                    stereo_distance.append(now_distance)
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3:]
                        ptsv_indexes[i] = voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]
                        near_align_num += 1
                        point_align_mask[i] = True
                    else:
                        point_rp_labels[i] = 0
                        ptsv_indexes[i] = -1
                        unalign_num += 1

            points[:, 3] = point_rp_labels
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            # mfe
            voxel_mean = np.sum(voxels, axis=1)
            normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            voxel_mean = voxel_mean / normalizer
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0

            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    point_mfe_rp_labels[i] = voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3:]
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                # point_rp_labels = point_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsv_indexes = ptsv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsv_indexes'] = ptsv_indexes

        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # compress
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0

            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points

            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])

        return data_dict

    # for mmdv3
    def transform_points_to_voxels_compress_mmdv3(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_compress_mmdv3, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        limit_range = self.point_cloud_range

        points = data_dict['points']
        # print("@@@@@@@@@@@@ points:", points)
        # print("@@@@@@@@@@@@ points:", points.shape) # (15072, 4)
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        points = points[mask]
        points_copy = copy.deepcopy(points)

        data_dict['points'] = points
        data_dict['points_copy'] = points_copy
        # print("@@@@@@@@@@@@@@@@@@@@ points_copy_size:", data_dict['points_copy'].shape)
        data_dict['point_num'] = points.shape[0]

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            # print("@@@@@@@@@@@@@@ done2") # no
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points # as student

        if self.mode == 'train':
            # second points - ori
            s_points = data_dict['s_points']
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) # (19069, 4)
            s_mask = (s_points[:, 0] >= limit_range[0]) & (s_points[:, 0] < limit_range[3]) \
                     & (s_points[:, 1] >= limit_range[1]) & (s_points[:, 1] < limit_range[4]) \
                     & (s_points[:, 2] >= limit_range[2]) & (s_points[:, 2] < limit_range[5])
            s_points = s_points[s_mask]
            # print("@@@@@@@@@@@@ s_points:", s_points.shape) #  (18487, 4)
            s_voxel_output = self.voxel_generator.generate(s_points)
            s_voxels, s_coordinates, s_num_points = s_voxel_output
            if not data_dict['use_lead_xyz']:
                s_voxels = s_voxels[..., 3:]  # remove xyz in voxels(N, 3)

            # mfe second points - ori
            s_voxel_mean = np.sum(s_voxels, axis=1)
            normalizer = np.clip(s_num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            s_voxel_mean = s_voxel_mean / normalizer

            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)
            # print('@@@@@@@@@@@@@@ voxel_hash_index:', voxel_hash_index) # [   0    1    2 ... 90111997 90111998 90111999]

            # ori point - voxels
            coords = np.array(s_coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(s_voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            # for rp_nr
            ptv_coordz = np.floor((s_points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((s_points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((s_points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0

            sptsv_indexes = np.zeros(s_points.shape[0], dtype=int)  # initialize 0
            sptsv_positive_mask = (sptsv_indexes == 1)  # initialize false
            for i in range(len(s_points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    # spoint_mfe_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3:]
                    sptsv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    sptsv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                s_points = s_points[sptsv_positive_mask]
                # spoint_mfe_rp_labels = spoint_mfe_rp_labels[sptsv_positive_mask]
                sptsv_indexes = sptsv_indexes[sptsv_positive_mask]

            # data_dict['point_mfe_rp_labels_tea2'] = spoint_mfe_rp_labels
            # for tea2
            s_points[:, 3] = 0
            data_dict['s_points2'] = s_points
            s_voxels[:, :, 3] = 0
            data_dict['voxels_tea2'] = s_voxels
            data_dict['voxel_coords_tea2'] = s_coordinates
            data_dict['voxel_num_points_tea2'] = s_num_points # as teacher
            data_dict['ptv_indexes_tea2'] = sptsv_indexes
            # for tea
            data_dict['s_points'] = s_points
            data_dict['s_point_num'] = s_points.shape[0]
            data_dict['voxels_tea'] = s_voxels
            data_dict['voxel_coords_tea'] = s_coordinates
            data_dict['voxel_num_points_tea'] = s_num_points # as teacher
            data_dict['ptv_indexes_tea'] = sptsv_indexes
            # print("@@@@@@@@@@@@@@@@ data_dict:", data_dict)

            # main points - compress
            # assign s_voxel_mean intensity to points
            point_rp_labels = np.zeros(points.shape[0], dtype=float) # initialize 0
            ptsv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            # point_align_mask = (point_rp_labels == 1) # initialize false
            # assign s_voxel_mean intensity to points
            direct_align_num = 0
            near_align_num = 0
            unalign_num = 0
            limit_xcoord = [0, x_num - 1]
            limit_ycoord = [0, y_num - 1]
            limit_zcoord = [0, z_num - 1]

            for i in range(len(points)):
            # for i in range(3):
                now_coords = [0, 0, 0]
                # nz, ny, nx
                now_coords[0] = (points[i][2] - limit_range[2]) / config.VOXEL_SIZE[2]
                now_coords[1] = (points[i][1] - limit_range[1]) / config.VOXEL_SIZE[1]
                now_coords[2] = (points[i][0] - limit_range[0]) / config.VOXEL_SIZE[0]
                now_coords = np.floor(now_coords).astype(int)
                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                if voxel_hash_ori_bool[now_product]:
                    point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[now_product]][3]
                    ptsv_indexes[i] = voxel_hash_ori_index[now_product]
                    direct_align_num += 1
                    # point_align_mask[i] = True
                else:
                    z_coords = np.clip([now_coords[0] - 1, now_coords[0], now_coords[0] + 1], limit_zcoord[0], limit_zcoord[1])
                    y_coords = np.clip([now_coords[1] - 1, now_coords[1], now_coords[1] + 1], limit_ycoord[0], limit_ycoord[1])
                    x_coords = np.clip([now_coords[2] - 1, now_coords[2], now_coords[2] + 1], limit_xcoord[0], limit_xcoord[1])
                    stereo_voxel_product = []
                    stereo_distance = []
                    for iz in range(3):
                        now_coords[0] = z_coords[iz]
                        for iy in range(3):
                            now_coords[1] = y_coords[iy]
                            for ix in range(3):
                                now_coords[2] = x_coords[ix]
                                now_product = now_coords[0] + now_coords[1] * z_num + now_coords[2] * z_num * y_num
                                if voxel_hash_ori_bool[now_product]:
                                    now_voxel_center = np.array([0.00, 0.00, 0.00])
                                    now_voxel_center[0] = limit_range[0] + (now_coords[2] + 0.5) * config.VOXEL_SIZE[0]
                                    now_voxel_center[1] = limit_range[1] + (now_coords[1] + 0.5) * config.VOXEL_SIZE[1]
                                    now_voxel_center[2] = limit_range[2] + (now_coords[0] + 0.5) * config.VOXEL_SIZE[2]
                                    # print("@@@@@@@@@@@@ points[i]:", points[i][0:3])
                                    # print("@@@@@@@@@@@@ now_coords:", now_coords)
                                    # print("@@@@@@@@@@@@ now_voxel_center:", now_voxel_center)
                                    now_distance = np.sqrt(sum(np.power((points[i][0:3] - now_voxel_center), 2)))
                                    # print("@@@@@@@@@@@@ now_distance:", i, " ", now_distance)
                                    # print("@@@@@@@@@@@@ s_voxel_mean:", i, " ", s_voxel_mean[voxel_hash_ori_index[now_product]][3:])
                                    stereo_voxel_product.append(now_product)
                                    stereo_distance.append(now_distance)
                                else:
                                    stereo_voxel_product.append(-1)
                                    stereo_distance.append(100)
                    if np.max(stereo_voxel_product) >= 0:
                        min_distance_idx = np.argmin(np.array(stereo_distance))
                        point_rp_labels[i] = s_voxel_mean[voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]][3]
                        ptsv_indexes[i] = voxel_hash_ori_index[stereo_voxel_product[min_distance_idx]]
                        near_align_num += 1
                        # point_align_mask[i] = True
                    else:
                        point_rp_labels[i] = 0
                        ptsv_indexes[i] = -1
                        unalign_num += 1

            points[:, 3] = point_rp_labels
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            # mfe
            voxel_mean = np.sum(voxels, axis=1)
            normalizer = np.clip(num_points, 1, config.MAX_POINTS_PER_VOXEL).reshape(-1, 1)
            voxel_mean = voxel_mean / normalizer
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0
            point_mfe_rp_labels = np.zeros(points.shape[0], dtype=float)  # initialize 0

            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    point_mfe_rp_labels[i] = voxel_mean[voxel_hash_ori_index[ptv_product[i]]][3]
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points_copy[ptv_positive_mask]
                point_mfe_rp_labels = point_mfe_rp_labels[ptv_positive_mask]
                # point_rp_labels = point_rp_labels[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                ptsv_indexes = ptsv_indexes[ptv_positive_mask]
                data_dict['points'] = points
            else:
                data_dict['points'] = points_copy

            data_dict['ptv_indexes'] = ptv_indexes
            data_dict['point_mfe_rp_labels'] = point_mfe_rp_labels
            data_dict['ptsv_indexes'] = ptsv_indexes

        else:
            x_num = round((limit_range[3] - limit_range[0]) / config.VOXEL_SIZE[0])
            y_num = round((limit_range[4] - limit_range[1]) / config.VOXEL_SIZE[1])
            z_num = round((limit_range[5] - limit_range[2]) / config.VOXEL_SIZE[2])
            total_num = x_num * y_num * z_num
            voxel_hash_index = np.arange(0, total_num, 1)

            # compress
            coords = np.array(coordinates)
            voxel_coords_product = coords[:, 0] + coords[:, 1] * z_num + coords[:, 2] * z_num * y_num
            voxel_index = np.arange(0, len(voxels), 1)
            voxel_hash_ori_bool = (voxel_hash_index == -1)
            voxel_hash_ori_index = np.ones(total_num, dtype=int) * -1
            voxel_hash_ori_bool[voxel_coords_product] = True
            voxel_hash_ori_index[voxel_coords_product] = voxel_index

            ptv_coordz = np.floor((points[:, 2] - limit_range[2]) / config.VOXEL_SIZE[2]).astype(int)
            ptv_coordy = np.floor((points[:, 1] - limit_range[1]) / config.VOXEL_SIZE[1]).astype(int)
            ptv_coordx = np.floor((points[:, 0] - limit_range[0]) / config.VOXEL_SIZE[0]).astype(int)
            ptv_product = ptv_coordz + ptv_coordy * z_num + ptv_coordx * z_num * y_num

            ptv_successnum = 0
            ptv_failnum = 0

            ptv_indexes = np.zeros(points.shape[0], dtype=int)  # initialize 0
            ptv_positive_mask = (ptv_indexes == 1) # initialize false
            for i in range(len(points)):
                if voxel_hash_ori_bool[ptv_product[i]]:
                    ptv_indexes[i] = voxel_hash_ori_index[ptv_product[i]]
                    ptv_positive_mask[i] = True
                    ptv_successnum += 1
                else:
                    ptv_failnum += 1

            if ptv_failnum > 0:
                points = points[ptv_positive_mask]
                ptv_indexes = ptv_indexes[ptv_positive_mask]
                data_dict['points'] = points

            data_dict['ptv_indexes'] = ptv_indexes

        # a = {}
        # print("stop:", a[10])

        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
