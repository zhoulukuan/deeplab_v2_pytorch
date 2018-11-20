from collections import OrderedDict

from torch import nn
import torch

from models.resnet import *
from utils.config import cfg

class Classifier(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class DeepLab(nn.Module):
    """"Deeplab for semantic segmentation """
    def __init__(self, num_classes, pretrained=True):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        self.pretrained = True
        self.pretrained_model = cfg.TRAIN.PRETRAINED_MODEL

    def _init_module(self):
        resnet = resnet101()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.pretrained_model))
            state_dict = torch.load(self.pretrained_model)
            temp = OrderedDict()
            for k, v in state_dict.items():
                new_key = k[6:]
                temp[new_key] = v
            resnet.load_state_dict({k: v for k, v in temp.items() if k in resnet.state_dict()})

        # Build base feature extractor
        self.ResNet_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.Pred_layer = Classifier([6, 12, 18, 24], [6, 12, 18, 24], self.num_classes)

        # Fix BatchNorm
        # No BN in pred_layer
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        self.ResNet_base.apply(set_bn_fix)

        # Fix blocks
        # for p in self.ResNet_base[i].parameters(): p.requires_grad = False

    def forward(self, *input):
        pass

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

        for m in self.Pred_layer.conv2d_list:
            normal_init(m, 0, 0.01)

    def create_architecture(self):
        self._init_module()
        self._init_weights()

    def train(self):
        pass

