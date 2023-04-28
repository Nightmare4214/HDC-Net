import logging
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import NibabelWriter
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils import MetricReduction


def get_wt_et_tc(x):
    """
    x: (H, W, T)
    """
    # WT = NET + ED + ET
    wt = (x > 0)
    # TC = NET + ET
    tc = (x == 1) | (x == 3)
    et = (x == 3)

    # return np.stack([wt, tc, et], axis=0)[None, ...]  # (1, 3, H, W, T)
    return torch.stack([wt, tc, et], dim=0)[None, ...]  # (1, 3, H, W, T)


def computational_runtime(runtimes):
    if not runtimes:
        return
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    if runtimes.shape[0] <= 2:
        meanTime = np.mean(runtimes)
    else:
        maxvalue = np.max(runtimes)
        minvalue = np.min(runtimes)
        nums = runtimes.shape[0] - 2
        meanTime = (np.sum(runtimes) - maxvalue - minvalue) / nums
    fps = 1 / meanTime
    print('mean runtime:', meanTime, 'fps:', fps)


# keys = 'whole', 'core', 'enhancing', 'loss'
keys = ['WT', 'TC', 'ET']


def validate_softmax(
        valid_loader,
        model,
        cfg='',
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        scoring=True,  # If true, print the dice score.
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        postprocess=False,  # Defualt False, when use postprocess, the score of dice_ET would be changed.
        cpu_only=False):
    assert cfg is not None
    H, W, T = 240, 240, 155
    model.eval()
    runtimes = []
    dice_metric = DiceMetric(reduction=MetricReduction.MEAN_BATCH) if scoring else None
    hausdorff = HausdorffDistanceMetric(include_background=True, percentile=95.,
                                        reduction=MetricReduction.MEAN_BATCH) if scoring else None
    writer = NibabelWriter(output_dtype=np.uint8) if savepath else None

    for idx, data in enumerate(valid_loader):
        x = data['image']
        meta = x.meta
        x = x.as_tensor().cuda(non_blocking=True)  # (1, 1, H, W, T)
        target = data['seg'].squeeze() if scoring else None  # (H, W, T)

        # compute output
        if not use_TTA:

            # torch.cuda.synchronize()
            start_time = time.time()
            logit = model(x)
            # torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            runtimes.append(elapsed_time)

            # output = F.softmax(logit, dim=1)
            output = logit
        else:
            start_time = time.time()
            logit = model(x)  # 000
            logit += model(x.flip(dims=(2,))).flip(dims=(2,))
            logit += model(x.flip(dims=(3,))).flip(dims=(3,))
            logit += model(x.flip(dims=(4,))).flip(dims=(4,))
            logit += model(x.flip(dims=(2, 3))).flip(dims=(2, 3))
            logit += model(x.flip(dims=(2, 4))).flip(dims=(2, 4))
            logit += model(x.flip(dims=(3, 4))).flip(dims=(3, 4))
            logit += model(x.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4))
            elapsed_time = time.time() - start_time
            runtimes.append(elapsed_time)

            output = logit / 8.0  # mean

        output = output[..., :H, :W, :T].argmax(dim=1).squeeze().cpu()  # (H, W, T)

        if postprocess:
            ET_mask = (output == 3)
            if ET_mask.sum() < 500:
                output[ET_mask] = 1

        msg = 'Subject {}/{}, '.format(idx + 1, len(valid_loader))
        name = str(idx)
        if names:
            name = names[idx]
            msg += '{:>20}, '.format(name)

        if scoring:
            output_wt_et_tc = get_wt_et_tc(output)  # (1, 3, height,width,depth)
            target_wt_et_tc = get_wt_et_tc(target)  # (1, 3, height,width,depth)
            dice = dice_metric(output_wt_et_tc, target_wt_et_tc)
            hausdorff(output_wt_et_tc, target_wt_et_tc)

            msg += ','.join(['{}_dice: {:4f}'.format(k, v) for k, v in zip(keys, dice.squeeze(0).tolist())])

            if snapshot:
                gap_width = 2
                Snapshot_img = torch.zeros(H, W * 2 + gap_width, T, 3, dtype=torch.uint8)
                output_one_hot = F.one_hot(output, num_classes=4)
                target_one_hot = F.one_hot(target, num_classes=4)
                Snapshot_img[:, :W, ...] = output_one_hot[..., 1:]
                Snapshot_img[:, W:W + gap_width, ...] = 1
                Snapshot_img[:, W + gap_width:, ...] = target_one_hot[..., 1:]
                Snapshot_img *= 255
                Snapshot_img = Snapshot_img.numpy()

                os.makedirs(os.path.join('snapshot', cfg, name), exist_ok=True)
                for frame in range(T):
                    imageio.imwrite(os.path.join('snapshot', cfg, name, str(frame) + '.png'),
                                    Snapshot_img[..., frame, :])

        if savepath:
            # .npy for farthur model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            elif save_format == 'nii':
                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = F.one_hot(output, num_classes=4)[..., 1:].numpy().astype(np.uint8) * 255
                    for frame in range(T):
                        os.makedirs(os.path.join(savepath, 'snapshot', name), exist_ok=True)
                        imageio.imwrite(os.path.join(savepath, 'snapshot', name, str(frame) + '.png'),
                                        Snapshot_img[..., frame, :])
                output[output == 3] = 4
                writer.set_data_array(output, channel_dim=None)
                writer.set_metadata({
                    'spatial_shape': meta['spatial_shape'],
                    'affine': meta['original_affine'],
                    'original_affine': meta['original_affine']
                }, resample=False, mode='nearest')
                writer.write(os.path.join(savepath, 'submission', name + '.nii.gz'))

                if verbose:
                    print('1:', torch.sum(output == 1).item(), ' | 2:', torch.sum(output == 2).item(), ' | 4:',
                          torch.sum(output == 4).item())
                    print('WT:', torch.sum((output == 1) | (output == 2) | (output == 4)).item(), ' | TC:',
                          torch.sum((output == 1) | (output == 4)).item(), ' | ET:', torch.sum(output == 4).item())
        logging.info(msg)

    if scoring:
        msg = 'Average scores:'
        msg += ','.join(['{}_dice: {:4f}'.format(k, v) for k, v in zip(keys, dice_metric.aggregate().tolist())])
        logging.info(msg)

        msg = 'Average scores:'
        msg += ','.join(['{}_HD95: {:4f}'.format(k, v) for k, v in zip(keys, hausdorff.aggregate().tolist())])
        logging.info(msg)

    computational_runtime(runtimes)

    # model.train()
    if dice_metric:
        return dice_metric.aggregate().tolist()
    return None
