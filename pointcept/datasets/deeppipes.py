"""
DeepPipes Dataset

get sampled point clouds of DeepPipes Dataset (XYZ from mesh, 100k points per shape)
at "https://github.com/wzsdu/DeepPipes_Dataset"

Author: wzsdu (https://github.com/wzsdu)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob


from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose
from .defaults import DefaultDataset


def resample_pcd(pcd, n):
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]], idx[:n]


@DATASETS.register_module()
class DeepPipesDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/deeppipes",
        class_names=None,
        transform=None,
        num_points=100000,
        uniform_sampling=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        if isinstance(class_names, tuple):
            class_names = class_names[0]
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        ) # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(self.data_root, self.split)
        data_list = glob.glob(os.path.join(split_path, "*"))
        return data_list
    
    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        name = os.path.basename(data_name)
        
        data_dict = {}
        data_dict['name'] = name
        data_dict['split'] = self.split
        coord = np.loadtxt(os.path.join(data_name, 'coord.pts'), dtype=np.float32)
        segment = np.loadtxt(os.path.join(data_name, 'label.seg'))[:,3].astype(np.int8)

        coord, idx = resample_pcd(coord, self.num_point)
        segment = segment[idx]

        data_dict['coord'] = coord
        data_dict['segment'] = segment

        return data_dict

    def __len__(self):
        return len(self.data_list) * self.loop