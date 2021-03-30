import torch
from torch.nn.utils.rnn import pad_sequence
from fastai.text.all import TfmdDL


class MySortedDL(TfmdDL):
    def __init__(self, dataset, pad_idx, srtkey_fc=None, filter_fc=False, cache_file=None, tfmd_args=None):
        """
        dataset: HF_Dataset Actually any object implements __len__ and __getitem__ that return a tuple as a sample.
        """
        pad_idxs = [pad_idx] * len(dataset[0])

        # Save attributes
        super().__init__(dataset, **tfmd_args)
        self.pad_idxs = pad_idxs

    def create_item(self, i):
        return self.dataset[i]

    def create_batch(self, samples):
        # if self.pad_idx is False: return super().create_batch(samples)
        pad_idxs = self.pad_idxs
        return tuple(
            pad_sequence(attr, batch_first=True,
                         padding_value=pad_idxs[i]) if attr[0].shape else torch.stack(attr)
            for i, attr in enumerate(zip(*samples)))
