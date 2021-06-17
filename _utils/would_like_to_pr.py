from statistics import mean
import torch
from torch import nn

"""
I would like more uniform way to pass the metrics, no matter loss_func or metric,
instantiate it and then pass.
This uniform way also make it possible such as `metrics=[m() for m inTASK_METRICS[task]]`
"""


def Accuracy(axis=-1):
    return AvgMetric(partial(accuracy, axis=axis))


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
