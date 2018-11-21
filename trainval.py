import os.path as osp
import os
import argparse
import pprint
import datetime
import re

import torch
from torch.utils.data import DataLoader
from torch import nn

import _init_path
from utils.config import cfg, cfg_from_file, cfg_from_list
from utils.tools import random_scale_and_msc, msc_label, adjust_learning_rate
from datasets.voc_loader import VOCDataset
from models.deeplab import DeepLab
from models.losses import loss_calc



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a deeplab network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
    parser.add_argument('--resume', dest='resume',
                      help='If resume training', default=False,
                      type=bool)
    parser.add_argument('--log_dir', dest='log_dir',
                      help='directory to save logs', default='logs',
                      type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='if use tensorboardX', default=True,
                      type=bool)
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
    valloader = DataLoader(dataset=valset, batch_size=1,
                           shuffle=False)

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
    if cfg.CUDA: net.cuda()

    data_iter = iter(trainloader)
    star_iter = 0
    ### If resume training
    if args.resume:
        load_model = sorted(os.listdir(args.save_dir))[-1]
        checkpoint = torch.load(osp.join(args.save_dir, load_model))
        lr = checkpoint['lr']
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        star_iter = checkpoint['iter']
        print("load model for %s !" % load_model)

    ### Use tensorboardX
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    ### Begin Training
    for iter in range(star_iter, cfg.TRAIN.MAX_ITERS+1):
        lr = adjust_learning_rate(optimizer, iter, lr)
        image, label = data_iter.next()
        img, img_75, img_50 = random_scale_and_msc(image.numpy(), label.numpy(), cfg.TRAIN.FIXED_SCALES, cfg.TRAIN.SCALES)
        if cfg.CUDA:
            img, img_75, img_50 = img.cuda().float(), img_75.cuda().float(), img_50.cuda().float()
        else:
            img, img_75, img_50 = img.float(), img_75.float(), img_50.float()

        out = net(img, img_75, img_50)

        label, label_75, label_50 = msc_label(label.numpy(), list(out[0].size()), list(out[1].size()), list(out[2].size()))
        if cfg.CUDA:
            label, label_75, label_50 = label.cuda().long(), label_75.cuda().long(), label_50.cuda().long()
        else:
            label, label_75, label_50 = label.long(), label_75.long(), label_50.long()

        loss = loss_calc(out, [label, label_75, label_50])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % args.disp_interval == 0:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(("Time:%s  [iter %4d/%4d]  loss: %.4f,  lr: %.2e" % (now, iter, cfg.TRAIN.MAX_ITERS, loss.item(), lr)))
            if args.use_tfboard:
                info = {
                    'loss': loss.item(),
                    'lr': lr
                }
                logger.add_scalars("logs/loss_lr", info, iter)

        if iter % args.checkpoint_interval == 0 or iter == cfg.TRAIN.MAX_ITERS:
            save_path = osp.join(args.save_dir, 'VOC12_'+str(iter)+'.pth')
            torch.save({'optimizer': optimizer.state_dict(),
                        'model': net.state_dict(),
                        'iter': iter,
                        'lr': lr
                        }, save_path)
            print("save model: {}".format(save_path))



    if args.use_tfboard:
        logger.close()



