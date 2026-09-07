import glob
import os
import numpy as np

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class GraspNetDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspNetDataset, self).__init__(**kwargs)
        self.grasp_files = glob.glob(os.path.join(file_path, 'rect_labels', 'scene_00*', "realsense", '*.npy'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('/rect_labels/', '/').replace('realsense/', 'realsense/depth/').replace('npy', 'png') for f in self.grasp_files]
        self.rgb_files = [f.replace('depth', 'rgb') for f in self.depth_files]

    def _get_crop_attrs(self, idx):
        img = image.Image.from_file(self.rgb_files[idx])
        center = np.array([img.shape[0] // 2, img.shape[1] // 2])
        ms = min(img.shape[0], img.shape[1])
        left = (img.shape[1] - ms) // 2
        top = (img.shape[0] - ms) // 2
        return center, left, top, ms

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center, left, top, ms = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        gtbbs.scale(self.output_size / ms)
        delitem = []
        for i in range(gtbbs.len()):
            if gtbbs[i].center[0] > self.output_size or gtbbs[i].center[1] > self.output_size or gtbbs[i].center[0] <= 0 or gtbbs[i].center[1] <=0:
                delitem.append(i)

        for j, i in enumerate(delitem):
            gtbbs.delete(i - j)

        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_png(self.depth_files[idx])
        center, left, top, ms = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (top + ms, left + ms))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        ms = min(rgb_img.shape[0], rgb_img.shape[1])
        center, left, top, _ = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (top + ms, left + ms))
        # print("2", rgb_img.shape, (top, left), (top + ms, left + ms))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        # print("3", rgb_img.shape)
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
