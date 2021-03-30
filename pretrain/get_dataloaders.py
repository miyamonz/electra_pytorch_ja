from ._utils.hf_dataset import HF_Dataset
from ._utils.mysorteddl import MySortedDL
from pathlib import Path
from fastai.text.all import TensorText


def get_dataloader(c, hf_tokenizer, train_dset, device='cpu'):
    print('train_dset', train_dset)
    args = {
        'cols': {'input_ids': TensorText, 'sentA_length': lambda x: x},
        'hf_toker': hf_tokenizer,
        'n_inp': 2,
    }
    ds = HF_Dataset(train_dset, **args)
    pad_idx = hf_tokenizer.pad_token_id
    tfmd_args = {
        'bs': c.bs,
        'num_workers': c.num_workers,
        'pin_memory': False,
        'shuffle': True,
        'drop_last': False,
        'device': device,
    }
    dl = MySortedDL(ds, pad_idx, srtkey_fc=False, tfmd_args=tfmd_args)
    return dl


# currently not used because MySortedDL doesn't have cache target when srtkey_fc = False
# cache_file = get_cache_file('./datasets/electra_dataloader', 'dl_{split}.json')
def get_cache_file(cache_dir, cache_name, split='train'):
    assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if not cache_name.endswith('.json'):
        cache_name += '.json'
    cache_file = cache_dir / cache_name.format(split=split)
    return cache_file
