#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from glob import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

columns = ['imgname',
           'x', 'y', 'z',  # 240, 240, 155
           'spacing_x', 'spacing_y', 'spacing_z',  # 1, 1, 1
           'origin_x', 'origin_y', 'origin_z',  # 0, -239, 0
           'direction_11', 'direction_12', 'direction_13',  # 1, 0, 0
           'direction_21', 'direction_22', 'direction_23',  # 0, 1, 0
           'direction_31', 'direction_32', 'direction_33']  # 0, 0, 1

columns2 = columns + ['labels']
columns.append('min')
columns.append('max')
modalities = ('flair', 't1ce', 't1', 't2')

train_path = '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training'
val_path = '/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Validation'

if __name__ == '__main__':
    df = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame(columns=columns2)
    files = list(glob(os.path.join(train_path, '*', '*', '*')))
    files.sort()

    for idx, file in enumerate(tqdm(files)):
        temp = [os.path.basename(file)]
        img = sitk.ReadImage(file)
        temp.extend(img.GetSize())
        temp.extend(img.GetSpacing())
        temp.extend(img.GetOrigin())
        temp.extend(img.GetDirection())
        img = sitk.GetArrayFromImage(img)
        if idx % 5 == 1:  # seg
            temp.append(','.join(map(str, np.unique(img))))
            df2 = pd.concat([df2, pd.DataFrame.from_records([temp], columns=columns2)], ignore_index=True)
        else:  # flair, t1ce, t1, t2
            temp.append(img.min())
            temp.append(img.max())
            df = pd.concat([df, pd.DataFrame.from_records([temp], columns=columns)], ignore_index=True)
    df.to_csv(os.path.join(train_path, 'image.csv'), index=False)
    df2.to_csv(os.path.join(train_path, 'label.csv'), index=False)

    df = pd.DataFrame(columns=columns)
    files = list(glob(os.path.join(val_path, '*', '*')))
    files.sort()

    for idx, file in enumerate(tqdm(files)):
        temp = [os.path.basename(file)]
        img = sitk.ReadImage(file)
        temp.extend(img.GetSize())
        temp.extend(img.GetSpacing())
        temp.extend(img.GetOrigin())
        temp.extend(img.GetDirection())
        img = sitk.GetArrayFromImage(img)

        temp.append(img.min())
        temp.append(img.max())
        df = pd.concat([df, pd.DataFrame.from_records([temp], columns=columns)], ignore_index=True)
    df.to_csv(os.path.join(val_path, 'val.csv'), index=False)
