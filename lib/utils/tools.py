import math
import random
import cv2
import numpy as np

import torch

def compute_outsize(im):
    """
    Compute the output size of input images
    stride: 8
    """
    h, w = im.shape[:2]
    for i in range(3):
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)
    return h, w

def random_scale_and_msc(image, lbl, fixed_scales, scales):
    """
    Random scale for data augmentation and get three fixed_scales for fuse scores
    """
    factor = random.uniform(scales[0], scales[1])
    factor = 1

    # img
    h, w = image.shape[1:3]
    img = [cv2.resize(temp, (int(h * factor), int(w * factor)))
           for temp in image[:]]
    img_75 = [cv2.resize(temp, (int(h * factor * 0.75), int(w * factor * 0.75)))
           for temp in image[:]]
    img_50 = [cv2.resize(temp, (int(h * factor * 0.5), int(w * factor * 0.5)))
           for temp in image[:]]

    # label
    h, w = compute_outsize(img[0])
    label = [cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]

    # change (B, H, W, C) to (B, C, H, W)
    # add label channel
    img = torch.from_numpy(np.array(img).transpose(0, 3, 1, 2))
    img_75 = torch.from_numpy(np.array(img_75).transpose(0, 3, 1, 2))
    img_50 = torch.from_numpy(np.array(img_50).transpose(0, 3, 1, 2))
    label = np.array(label)
    label = label[:, :, :, np.newaxis]
    label = torch.from_numpy(label.transpose(0, 3, 1, 2))


    return img, img_75, img_50, label


def read_from_model(path):
    pass