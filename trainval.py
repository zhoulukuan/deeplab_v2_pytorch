import os.path as osp
import argparse
import pprint

import torch
from torch.utils.data import DataLoader
from torch import nn

import _init_path
from utils.config import cfg, cfg_from_file, cfg_from_list
from utils.tools import random_scale_and_msc
from datasets.voc_loader import VOCDataset
from models.deeplab import DeepLab



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print('Called with args: ')
    print(args)

    args.cfg_file = "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # torch.backends.cudnn.benchmark = True # Due to the different sizes of VOC images
    if torch.cuda.is_available() and not cfg.CUDA:
        print("Warning: You have a CUDA device, so you should run on it")

    if args.dataset == 'pascal_voc':
        num_classes = 21
        dataset = VOCDataset(osp.join(cfg.DATA_DIR, 'train.txt') ,cfg, num_classes)
        valset = VOCDataset(osp.join(cfg.DATA_DIR, 'val.txt'), cfg, num_classes)

    trainloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                             shuffle=True)
    valloader = DataLoader(dataset=dataset, batch_size=1,
                           shuffle=True)

    net = DeepLab(num_classes, pretrained=False)
    net.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    momentum = cfg.TRAIN.MOMENTUM

    ### Setting learning rate
    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'ResNet_base' in key:
                # I don't set bias weight decay to zero
                params += [{'params': [value], 'lr': lr}]
            elif 'Pred_layer' in key:
                params += [{'params': [value], 'lr': 10 * lr}]
            else:
                raise Exception('Nonexistent layers! ')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, momentum=momentum, weight_decay=weight_decay)


    for iter in cfg.TRAIN.MAX_ITERS:
        pass