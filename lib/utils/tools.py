import math
import random
import cv2
import numpy as np

import torch

from utils.config import cfg

def random_scale_and_msc(image, lbl, fixed_scales, scales, aug=True):
    """
    Random scale for data augmentation and get three fixed_scales for fuse scores
    """
    if aug: factor = random.uniform(scales[0], scales[1])
    else: factor = 1

    # img
    h, w = image.shape[1:3]
    img = [cv2.resize(temp, (int(w * factor), int(h * factor)))
           for temp in image[:]]
    img_75 = [cv2.resize(temp, (int(w * factor * fixed_scales[1]), int(h * factor * fixed_scales[1])))
           for temp in image[:]]
    img_50 = [cv2.resize(temp, (int(w * factor * fixed_scales[0]), int(h * factor * fixed_scales[0])))
           for temp in image[:]]

    # change (B, H, W, C) to (B, C, H, W)
    img = torch.from_numpy(np.array(img).transpose(0, 3, 1, 2))
    img_75 = torch.from_numpy(np.array(img_75).transpose(0, 3, 1, 2))
    img_50 = torch.from_numpy(np.array(img_50).transpose(0, 3, 1, 2))


    return img, img_75, img_50


def msc_label(lbl, s1, s2, s3):
    lbl[lbl == 255] = 0

    label = [cv2.resize(temp, (s1[3], s1[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label = torch.from_numpy(np.array(label))

    label_75 = [cv2.resize(temp, (s2[3], s2[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label_75 = torch.from_numpy(np.array(label_75))

    label_50 = [cv2.resize(temp, (s3[3], s3[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label_50 = torch.from_numpy(np.array(label_50))

    return label, label_75, label_50


def adjust_learning_rate(optimizer, iter, lr):
    """
    Change learning rate in optimizer.
    Return learning rate of resnet part for tensorboardX
    """
    if not cfg.TRAIN.IF_POLY_POLICY and iter % cfg.TRAIN.LR_DECAY_ITERS or iter == 0: return lr
    for param_group in optimizer.param_groups:
        if cfg.TRAIN.IF_POLY_POLICY and iter != 0:
            s = math.pow((1 - iter / cfg.TRAIN.MAX_ITERS), cfg.TRAIN.POWER)
            t = round(param_group['lr'] / lr) # t = 1 or 10
            # assert (t == 1 or t == 10)
            if not(t == 1 or t == 10):
                print("\n\n Not 1 or 10 when iter = %d, t = %f \n\n" % (iter, param_group['lr'] / lr))
            param_group['lr'] = t * cfg.TRAIN.LEARNING_RATE * s
        elif iter % cfg.TRAIN.LR_DECAY_ITERS == 0 and iter != 0:
            param_group['lr'] = 0.1 * param_group['lr']

    if cfg.TRAIN.IF_POLY_POLICY:
        return cfg.TRAIN.LEARNING_RATE * s
    else:
        return 0.1 * lr



