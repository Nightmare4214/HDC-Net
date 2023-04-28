import os
import torch
from torch.utils.data import Dataset

# noinspection PyUnresolvedReferences
from .transforms import StackImagesd, PercentileAndZScored
# noinspection PyUnresolvedReferences
from monai.transforms import MapTransform, Compose, LoadImaged, CastToTyped, EnsureChannelFirstd, RandSpatialCropd, \
    RandRotated, RandScaleIntensityd, RandShiftIntensityd, RandFlipd, Pad, Padd

import numpy as np

modalities = ('flair', 't1ce', 't1', 't2')
label = 'seg'


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', find_label=True, transforms=''):
        paths = []
        names = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name = os.path.basename(line)
                names.append(name)

                temp = {}
                for modality in modalities:
                    temp[modality] = os.path.join(root, line, f'{name}_{modality}.nii.gz')
                if find_label:
                    temp['seg'] = os.path.join(root, line, f'{name}_seg.nii.gz')
                paths.append(temp)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        temp = self.transforms(self.paths[index])
        for k in modalities:
            temp.pop(k)

        if 'seg' in temp:
            temp['seg'][temp['seg'] == 4] = 3
        # {'image': (4, H, W, D),
        #  'seg': (1, H, W, D)}
        return temp

    def __len__(self):
        return len(self.names)
