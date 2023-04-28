# coding=utf-8
import argparse
import logging
import os

import torch
import torch.optim
from torch.utils.data import DataLoader

import models
from data import datasets
from predict import validate_softmax
from utils import Parser, str2bool
from utils.utils import setup_seed

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='HDC_Net', required=True, type=str,
                    help='Your detailed configuration of the network')

parser.add_argument('-mode', '--mode', default=1, required=True, type=int, choices=[0, 1, 2],
                    help='0 for cross-validation on the training set; '
                         '1 for validing on the validation set; '
                         '2 for testing on the testing set.')

parser.add_argument('-gpu', '--gpu', default='0', type=str)

parser.add_argument('-is_out', '--is_out', default=False, type=str2bool,
                    help='If ture, output the .nii file')

parser.add_argument('-verbose', '--verbose', default=True, type=str2bool,
                    help='If True, print more infomation of the debuging output')

parser.add_argument('-use_TTA', '--use_TTA', default=False, type=str2bool,
                    help='It is a postprocess approach.')

parser.add_argument('-postprocess', '--postprocess', default=False, type=str2bool,
                    help='Another postprocess approach.')

parser.add_argument('-save_format', '--save_format', default='nii', choices=['nii', 'npy'], type=str,
                    help='[nii] for submission; [npy] for models ensemble')

parser.add_argument('-snapshot', '--snapshot', default=False, type=str2bool,
                    help='If True, saving the snopshot figure of all samples.')

parser.add_argument('-restore', '--restore', default=argparse.SUPPRESS, type=str,
                    help='The path to restore the model.')  # 'model_epoch_300.pth'

path = os.path.dirname(__file__)

args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
args.gpu = str(args.gpu)
ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.restore)  # specify the epoch


def main():
    # setup environments and seeds
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    g = torch.Generator()
    g.manual_seed(0)
    setup_seed(args.seed)

    Network = getattr(models, args.net)  #
    model = Network(**args.net_params)

    model = torch.nn.DataParallel(model).cuda()
    print(args.resume)
    assert os.path.isfile(args.resume), "no checkpoint found at {}".format(args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))

    msg += '\n' + str(args)
    logging.info(msg)

    if args.mode == 0:
        root_path = args.train_data_dir
        is_scoring = True
        valid_list = os.path.join(root_path, args.valid_list)
    elif args.mode == 1:
        root_path = args.valid_data_dir
        is_scoring = False
        valid_list = os.path.join(root_path, args.valid_list)
    elif args.mode == 2:
        root_path = args.test_data_dir
        is_scoring = False
        valid_list = os.path.join(root_path, args.test_list)
    else:
        raise ValueError

    Dataset = getattr(datasets, args.dataset)  #
    valid_set = Dataset(valid_list, root=root_path, find_label=is_scoring, transforms=args.test_transforms)

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        # collate_fn=valid_set.collate,
        # num_workers=1,
        # pin_memory=False
    )

    if args.is_out:
        # out_dir = './output/{}'.format(args.cfg)
        out_dir = os.path.join('output', args.cfg)
        os.makedirs(os.path.join(out_dir, 'submission'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'snapshot'), exist_ok=True)
    else:
        out_dir = ''

    logging.info('-' * 50)
    logging.info(msg)

    with torch.no_grad():
        validate_softmax(
            valid_loader,
            model,
            cfg=args.cfg,
            savepath=out_dir,
            save_format=args.save_format,
            names=valid_set.names,
            scoring=is_scoring,
            verbose=args.verbose,
            use_TTA=args.use_TTA,
            snapshot=args.snapshot,
            postprocess=args.postprocess,
            cpu_only=False)


if __name__ == '__main__':
    main()
