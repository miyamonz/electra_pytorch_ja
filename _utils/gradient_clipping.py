from torch import nn
from fastai.text.all import Callback


class GradientClipping(Callback):
    def __init__(self, clip: float = 0.1):
        self.clip = clip
        assert self.clip

    def after_backward(self):
        if hasattr(self, "scaler"):
            self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
