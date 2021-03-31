import torch
from torch import nn
from fastai.text.all import delegates, BaseLoss


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
