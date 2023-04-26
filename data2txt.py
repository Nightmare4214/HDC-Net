#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from glob import glob
import argparse

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--root_path', type=str,
                        default='/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training', help='root path')
    parser.add_argument('-o', '--output_path', type=str,
                        default='train.txt', help='output txt')
    args = parser.parse_args()
    root_path = args.root_path
    dirs = list(glob(os.path.join(root_path, '*', '*')))
    if not dirs or 'HGG' not in dirs[0]:  # MICCAI_BraTS_2018_Data_validation
        dirs = list(glob(os.path.join(root_path, '*')))
    dirs = list(filter(lambda x: os.path.isdir(x), dirs))
    dirs.sort()

    with open(os.path.join(root_path, args.output_path), 'w') as f:
        for cur_dir in tqdm(dirs):
            f.write(os.path.relpath(cur_dir, root_path) + '\n')
