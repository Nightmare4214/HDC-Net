# coding=utf-8
import argparse
import os
import random

import monai
import numpy as np
import torch
import torch.optim
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import models
from data import datasets
from utils import Parser
from utils.utils import setup_seed, seed_worker

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='HDC_Net', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)  # model_last.pth

# parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
# args.net_params.device_ids= [int(x) for x in (args.gpu).split(',')]
ckpts = args.makedir()

args.resume = os.path.join(ckpts, args.restore)  # specify the epoch


def main():
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    g = torch.Generator()
    g.manual_seed(0)
    setup_seed(args.seed)

    Network = getattr(models, args.net)  #
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    optimizer.zero_grad()
    criterion = getattr(monai.losses, args.criterion)(**args.criterion_kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x * 1.0 / args.num_epochs) ** 0.9)

    # Data loading code
    Dataset = getattr(datasets, args.dataset)  #

    train_list = os.path.join(args.train_data_dir, args.train_list)
    train_set = Dataset(train_list, root=args.train_data_dir, find_label=True, transforms=args.train_transforms)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g
    )

    valid_list = os.path.join(args.train_data_dir, args.valid_list)
    valid_set = Dataset(valid_list,
                        root=args.train_data_dir,
                        find_label=True,
                        transforms=args.test_transforms)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=5,
        pin_memory=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g
    )

    best_loss = 1000
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_loss = checkpoint['best_loss']
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['random_state'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'
    msg += '\n' + str(args)
    print(msg)
    writer = SummaryWriter(log_dir=ckpts)

    with tqdm(range(args.start_iter, args.num_epochs), initial=args.start_iter, total=args.num_epochs) as pbar:
        for epoch in pbar:
            train_loss_average = AverageMeter()
            model.train()
            for idx, data in enumerate(train_loader):
                x = data['image'].as_tensor().cuda(non_blocking=True)  # (B,4,H,W,D)
                target = data['seg'].as_tensor().cuda(non_blocking=True)  # (B,1,H,W,D)

                output = model(x)
                cur_loss = criterion(output, target)
                train_loss_average.update(cur_loss.item(), x.size(0))
                del x
                del output
                del target
                torch.cuda.empty_cache()

                cur_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(
                    "epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, best:{val_loss:.6f}".format(
                        ep=epoch + 1, allep=args.num_epochs, num=idx + 1, batch_num=len(train_loader),
                        ls=cur_loss.item(),
                        val_loss=best_loss))
                writer.add_scalar('Loss/train', train_loss_average.avg, epoch)
            scheduler.step()

            if args.valid_freq > 0 and 0 == (epoch + 1) % args.valid_freq:
                with torch.no_grad():
                    val_loss_average = AverageMeter()
                    model.eval()
                    for idx, data in enumerate(valid_loader):
                        x = data['image'].as_tensor().cuda(non_blocking=True)  # (B,4,H,W,D)
                        target = data['seg'].as_tensor().cuda(non_blocking=True)  # (B,1,H,W,D)
                        H, W, D = target.shape[2:]
                        output = model(x)[..., :H, :W, :D]
                        cur_loss = criterion(output, target)
                        val_loss_average.update(cur_loss.item(), x.size(0))
                        del x
                        del output
                        del target
                        torch.cuda.empty_cache()
                    val_loss = val_loss_average.avg
                    writer.add_scalar('Loss/val', val_loss, epoch)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        file_name = os.path.join(ckpts, 'model_best.pth'.format(epoch))
                        torch.save({
                            'iter': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_loss': best_loss,
                            'torch_rng_state': torch.get_rng_state(),
                            'np_rng_state': np.random.get_state(),
                            'random_state': random.getstate()
                        }, file_name)
            if 0 == (epoch + 1) % args.save_freq:
                file_name = os.path.join(ckpts, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'iter': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'torch_rng_state': torch.get_rng_state(),
                    'np_rng_state': np.random.get_state(),
                    'random_state': random.getstate()
                }, file_name)

    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'iter': epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_loss': best_loss,
        'torch_rng_state': torch.get_rng_state(),
        'np_rng_state': np.random.get_state(),
        'random_state': random.getstate()
    }, file_name)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    main()
