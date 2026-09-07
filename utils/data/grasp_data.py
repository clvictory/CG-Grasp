import random

import numpy as np
import torch
import torch.utils.data
import math
from .graph import connected_graphs
from ..dataset_processing.grasp import GraspRectangles
from sklearn.cluster import KMeans

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, dist_n=4):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.dist_n = dist_n

        self.grasp_files = []
        self.wmax = 0.
        self.wmin = 300.

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s, dtype=np.float32):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(dtype))
        else:
            return torch.from_numpy(s.astype(dtype))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # gt = []
        # for i, gr in enumerate(bbs.grs):
        #     gt.append({'id': i, 'x': gr.center[0], 'y': gr.center[1], 'angle': gr.angle, 'width': gr.length})

        # bb_idx = connected_graphs(gt)
        # q_n_gr = []
        # for i in range(self.dist_n):
        #     b_idx = [i % len(bb_idx)]
        #     q_n_gr.append(GraspRectangles([bbs.grs[bi] for bi in b_idx]))
        # q_n_img = []
        # w_n_img = []
        # a_n_img = []
        # for q_gr in q_n_gr:
        #     pos_img, ang_img, width_img = q_gr.draw((self.output_size, self.output_size))
        #     mask = pos_img > 0.9
        #     q_n_img.append(mask)
        # q_n_img = np.array(q_n_img)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        # pos_mask = pos_img == 1.0

        # print("angle :", ang_img.max(), ang_img.min(), ang_img.shape)
        # print("angle data:", ang_img[0, :20])
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)
        # width_n_img = self.dis_n_stat(width_img, pos_mask=pos_mask)
        # sin_n_img = self.dis_n_stat(np.sin(ang_img * 2), pos_mask=pos_mask)
        # cos_n_img = self.dis_n_stat(np.cos(ang_img * 2), pos_mask=pos_mask)        

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        # q_n = self.numpy_to_torch(q_n_img, dtype=np.bool)
        # cos_n = self.numpy_to_torch(cos_n_img, dtype=np.bool)
        # sin_n = self.numpy_to_torch(sin_n_img, dtype=np.bool)
        # width_n = self.numpy_to_torch(width_n_img, dtype=np.bool)
        # print(q_n.shape, cos_n.shape, sin_n.shape, width_n.shape)
        # return x, (q_n, cos_n, sin_n, width_n), idx, rot, zoom_factor

        # if x.shape[0] == 4:
        #     print(self.grasp_files[idx])
        #     print(rgb_img.shape)
        #     print(x.shape, pos.shape, cos.shape, sin.shape, width.shape)
        #     print(self.include_depth, self.include_rgb)
        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)

    def dis_n_stat(self, arr, pos_mask):
        results = []
        arr_t = arr[pos_mask]
        mask_indices = np.where(pos_mask)
        mask_indices = np.array(mask_indices).T.reshape(-1, 2)
        kmeans = KMeans(n_clusters=self.dist_n)  
        kmeans.fit(arr_t.reshape(-1, 1))
        label = kmeans.predict(arr_t.reshape(-1, 1))
            
        reconstructed_data = np.zeros_like(arr)  
        for mi, l in zip(mask_indices, label):
            reconstructed_data[mi[0], mi[1]] = (l + 1)
        dis_zero = []
        dis_nozero = []
        zeronum = 0
        for i in range(self.dist_n):
            mask = (reconstructed_data == (i + 1))
            mask = mask & pos_mask
            # dis_zero.append(arr[mask].shape)
            if arr[mask].shape[0] == 0:
                zeronum += 1
                dis_zero.append(i)
            else:
                dis_nozero.append(i)
            results.append(mask)
        # if zeronum > 0:
        #     dis_nozero = np.array(dis_nozero)
        #     dis_zero = np.array(dis_zero)
        #     pad_nozero = pad_array(dis_nozero, zeronum)
        #     for (zi, pi) in zip(dis_zero, pad_nozero):
        #         results[zi] = results[pi]
        return np.array(results)

def pad_array(arr, target_size):
    if arr.size < target_size:  
        padding = (target_size - arr.size) // 2  
        arr = np.pad(arr, (padding, padding), mode='edge')  
    elif arr.size > target_size:
        arr = arr[:target_size]  
    return arr  