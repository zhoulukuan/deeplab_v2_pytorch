
from torch import nn
import torch

from models.resnet import *
from utils.config import cfg

from collections import OrderedDict


class DeepLab(nn.Module):
    """"Deeplab for semantic segmentation """
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = cfg.TRAIN.PRETRAINED_MODEL

    def _init_module(self):
        # Define the network
        self.Scale = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=self.num_classes)

        # Fix BatchNorm
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        self.Scale.apply(set_bn_fix)

        # Fix blocks

    def forward(self, input, input_75, input_50):
        input_size = input.size()
        h, w = input.size()[2:]

        out = []
        x = self.Scale(input)
        out.append(x)
        interp = nn.UpsamplingBilinear2d(size=(x.size()[2], x.size()[3]))
        fuse = x

        x = self.Scale(input_75)
        out.append(interp(x))
        fuse = torch.max(fuse, interp(x))

        x = self.Scale(input_50)
        out.append(x)
        out.append(torch.max(fuse, interp(x)))

        return out

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        for m in self.Scale.layer5.conv2d_list:
            normal_init(m, 0, 0.01)

    def create_architecture(self):
        self._init_module()
        self._init_weights()


