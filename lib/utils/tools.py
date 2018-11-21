import math
import random
import cv2
import numpy as np

import torch

from utils.config import cfg

def random_scale_and_msc(image, lbl, fixed_scales, scales):
    """
    Random scale for data augmentation and get three fixed_scales for fuse scores
    """
    factor = random.uniform(scales[0], scales[1])

    # img
    h, w = image.shape[1:3]
    img = [cv2.resize(temp, (int(h * factor), int(w * factor)))
           for temp in image[:]]
    img_75 = [cv2.resize(temp, (int(h * factor * fixed_scales[1]), int(w * factor * fixed_scales[1])))
           for temp in image[:]]
    img_50 = [cv2.resize(temp, (int(h * factor * fixed_scales[0]), int(w * factor * fixed_scales[0])))
           for temp in image[:]]

    # label
    # h, w = compute_outsize(img[0])
    # label = [cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    # h, w = compute_outsize(img_75[0])
    # label_75 = [cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    # h, w = compute_outsize(img_50[0])
    # label_50 = [cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]


    # change (B, H, W, C) to (B, C, H, W)
    img = torch.from_numpy(np.array(img).transpose(0, 3, 1, 2))
    img_75 = torch.from_numpy(np.array(img_75).transpose(0, 3, 1, 2))
    img_50 = torch.from_numpy(np.array(img_50).transpose(0, 3, 1, 2))
    # label = np.array(label)
    # label = torch.from_numpy(label)
    # label = label[:, :, :, np.newaxis]
    # label = torch.from_numpy(label.transpose(0, 3, 1, 2))
    # label_75 = np.array(label_75)
    # label_75 = torch.from_numpy(label_75)
    # label_75 = label_75[:, :, :, np.newaxis]
    # label_75 = torch.from_numpy(label_75.transpose(0, 3, 1, 2))
    # label_50 = np.array(label_50)
    # label_50 = torch.from_numpy(label_50)
    # label_50 = label_50[:, :, :, np.newaxis]
    # label_50 = torch.from_numpy(label_50.transpose(0, 3, 1, 2))


    return img, img_75, img_50


def msc_label(lbl, s1, s2, s3):
    label = [cv2.resize(temp, (s1[2], s1[3]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label = torch.from_numpy(np.array(label))

    label_75 = [cv2.resize(temp, (s2[2], s2[3]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label_75 = torch.from_numpy(np.array(label_75))

    label_50 = [cv2.resize(temp, (s3[2], s3[3]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
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
            s1 = math.pow((1 - iter / cfg.TRAIN.MAX_ITERS), cfg.TRAIN.POWER)
            s2 = math.pow((1 - (iter - 1) / cfg.TRAIN.MAX_ITERS), cfg.TRAIN.POWER)
            param_group['lr'] = param_group['lr'] / s2 * s1
            return lr / s2 * s1
        elif iter % cfg.TRAIN.LR_DECAY_ITERS == 0 and iter != 0:
            param_group['lr'] = 0.1 * param_group['lr']
            return 0.1 * lr



