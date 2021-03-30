class HF_Dataset():

    def __init__(self, hf_dset, cols=None, hf_toker=None, n_inp=1):
        assert isinstance(cols, dict)

        # make dataset output pytorch tensor
        hf_dset.set_format(type='torch', columns=list(cols.keys()))

        # store attributes
        # self.pad_idx = hf_toker.pad_token_id
        self.hf_dset = hf_dset
        self.cols = cols
        self.n_inp = n_inp
        # self.hf_toker = hf_toker

    def __getitem__(self, idx):
        sample = self.hf_dset[idx]
        return tuple(tensor_cls(sample[col]) for col, tensor_cls in self.cols.items())

    def __len__(self):
        return len(self.hf_dset)

    # __getitem__, __len__以外はいらないはず
