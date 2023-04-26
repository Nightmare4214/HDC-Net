#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path',
                        default='/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/all.txt', type=str,
                        help='input txt')
    parser.add_argument('-k', '--k_fold', default=5, type=int, help='k fold')
    parser.add_argument('-o', '--one', default=1, type=int, help='k txt or train val')
    args = parser.parse_args()

    path = args.input_path

    root_path = os.path.dirname(path)
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                X.append(line)
                if 'HGG' in line:
                    Y.append(1)
                else:  # LGG
                    Y.append(0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    X = np.array(X)
    for idx, (train_index, val_index) in enumerate(skf.split(X, Y)):
        train_path = 'train.txt' if args.one else f'train_{idx}.txt'
        val_path = 'val.txt' if args.one else f'val{idx}.txt'
        with open(os.path.join(root_path, train_path), 'w') as f:
            f.write('\n'.join(X[train_index]))
        with open(os.path.join(root_path, val_path), 'w') as f:
            f.write('\n'.join(X[val_index]))
        if args.one:
            break
