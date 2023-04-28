#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import torch
import SimpleITK as sitk
from monai.config import KeysCollection
from monai.data import NibabelWriter
from monai.transforms import MapTransform, Compose, LoadImaged, CastToTyped, EnsureChannelFirstd, RandSpatialCropd, \
    RandRotated, RandScaleIntensityd, RandShiftIntensityd, RandFlipd, Pad, Padd

data = {
    'flair': '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz',
    'seg': '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz',
    't1': '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz',
    't1ce': '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1ce.nii.gz',
    't2': '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t2.nii.gz'
}


def check_nii_same(a, b):
    left = sitk.ReadImage(a)
    right = sitk.ReadImage(b)
    if not np.allclose(np.array(left.GetSize()), np.array(right.GetSize())):
        return False
    if left.GetPixelIDTypeAsString() == right.GetPixelIDTypeAsString():
        return False
    if not np.allclose(np.array(left.GetOrigin()), np.array(right.GetOrigin())):
        return False
    if not np.allclose(np.array(left.GetSpacing()), np.array(right.GetSpacing())):
        return False
    if not np.allclose(np.array(left.GetDirection()), np.array(right.GetDirection())):
        return False

    return True


def dict_equal(a: dict, b: dict, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = []
    if a.keys() != b.keys():
        return False
    try:
        for k in a.keys():
            if k in ignore_keys:
                continue
            left = a[k]
            right = b[k]
            if isinstance(left, torch.Tensor):
                if not (torch.allclose(left, right) or (torch.isnan(left) and torch.isnan(right))):
                    return False
            elif isinstance(left, np.ndarray):
                if not (np.allclose(left, right) or (np.isnan(left) and np.isnan(right))):
                    return False
            elif left != right:
                return False

    except:
        return False
    return True


class StackImagesd(MapTransform):
    """
    stack images
    add the result in the dict with the key 'image'
    """

    def __call__(self, data):
        d = dict(data)
        result = []
        for key in self.key_iterator(d):
            result.append(d[key])
        d['image'] = torch.stack(result, dim=0)  # (H, W, D)->(4, H, W, D)
        return d


class PercentileAndZScored(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            lower_percentile: float = 0.,
            upper_percentile: float = 100.,
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            lower_percentile: lower percentile(0-100)
            upper_percentile: upper percentile(0-100)
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.lower_percentile = lower_percentile / 100.
        self.upper_percentile = upper_percentile / 100.

    def __call__(self, data):
        # the first dim of data should be the channel(CHW[D])
        d = dict(data)
        for key in self.key_iterator(d):
            images = data[key]
            C = images.size(0)
            mask = images.sum(0) > 0  # brain
            for k in range(C):
                x = images[k, ...]
                y = x[mask]
                lower = torch.quantile(y, self.lower_percentile)
                upper = torch.quantile(y, self.upper_percentile)

                x[mask & (x < lower)] = lower
                x[mask & (x > upper)] = upper

                # z-score across the brain
                y = x[mask]
                x -= y.mean()
                x /= y.std()

                images[k, ...] = x
            d[key] = images
        return d


if __name__ == '__main__':
    transform = Compose([
        LoadImaged(['flair', 't1', 't1ce', 't2', 'seg'], image_only=True, allow_missing_keys=True),

        CastToTyped(keys=['seg'], dtype=torch.long, allow_missing_keys=True),
        EnsureChannelFirstd(keys=['seg'], allow_missing_keys=True),

        StackImagesd(keys=['flair', 't1', 't1ce', 't2']),  # add ['image']
        PercentileAndZScored(keys=['image'], lower_percentile=0.2, upper_percentile=99.8),

        RandSpatialCropd(keys=['image', 'seg'], roi_size=(128, 128, 128), random_size=False, allow_missing_keys=True),
        RandRotated(keys=['image', 'seg'], range_x=10, range_y=10, range_z=10, allow_missing_keys=True),  # (-10, 10)
        CastToTyped(keys=['seg'], dtype=torch.long, allow_missing_keys=True),
        RandScaleIntensityd(keys=['image'], factors=0.1),  # (-0.1, 0.1), img * (1 + scale)
        RandShiftIntensityd(keys=['image'], offsets=0.1),  # (-0.1, 0.1), img + offset
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
        CastToTyped(keys=['seg'], dtype=torch.long, allow_missing_keys=True),
        Padd(keys=["image", "seg"], padder=Pad([(0, 0), (0, 0), (0, 0), (0, 5)]), allow_missing_keys=True),
    ])
    result = transform(data)
    print(result['seg'].shape)
    image = torch.stack([result['image']], dim=0)
    mask = torch.randint(0, 4, (1, 240, 240, 155))
    mask[mask == 4] = 3
    import copy
    pre = copy.deepcopy(mask).squeeze()
    writer = NibabelWriter(output_dtype=torch.uint8)
    writer.set_data_array(mask)
    writer.set_metadata({
        'spatial_shape': result['image'].meta['spatial_shape'],
        'affine': result['image'].meta['original_affine'],
        'original_affine': result['image'].meta['original_affine']
    }, resample=False, mode='nearest')

    writer.write('faQ.nii.gz')
    data_obj = writer.data_obj
    temp = torch.from_numpy(data_obj.get_fdata())
    print(torch.allclose(pre.float(), temp.float()))

    print(check_nii_same('faQ.nii.gz', data['seg']))
