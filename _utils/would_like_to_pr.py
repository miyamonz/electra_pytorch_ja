from statistics import mean
import torch
from torch import nn
from fastai.text.all import *

"""
I would like more uniform way to pass the metrics, no matter loss_func or metric,
instantiate it and then pass.
This uniform way also make it possible such as `metrics=[m() for m inTASK_METRICS[task]]`
"""


def Accuracy(axis=-1):
    return AvgMetric(partial(accuracy, axis=axis))


@delegates()
class MyMSELossFlat(BaseLoss):
    def __init__(self, *args, axis=-1, floatify=True, low=None, high=None, **kwargs):
        super().__init__(
            nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs
        )
        self.low, self.high = low, high

    def decodes(self, x):
        if self.low is not None:
            x = torch.max(x, x.new_full(x.shape, self.low))
        if self.high is not None:
            x = torch.min(x, x.new_full(x.shape, self.high))
        return x


class Ensemble(nn.Module):
    def __init__(self, models, device="cuda:0", merge_out_fc=None):
        super().__init__()
        self.models = nn.ModuleList(m.cpu() for m in models)
        self.device = device
        self.merge_out_fc = merge_out_fc

    def to(self, device):
        self.device = device
        return self

    def getitem(self, i):
        return self.models[i]

    def forward(self, *args, **kwargs):
        outs = []
        for m in self.models:
            m.to(self.device)
            out = m(*args, **kwargs)
            m.cpu()
            outs.append(out)
        if self.merge_out_fc:
            outs = self.merge_out_fc(outs)
        else:
            outs = torch.stack(outs)
            outs = outs.mean(dim=0)
        return outs
