import torch
from torch import nn

def loss_calc(pred, label):
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()

    loss = criterion(m(pred[0]), label[0])
    label.append(label[0])
    for i in range(1, len(pred)):
        loss += criterion(m(pred[i]), label[i])

    return loss