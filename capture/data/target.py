import typing
from pathlib import Path
from PIL import Image
import random

import torch
from easydict import EasyDict
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset

from ..utils.log import get_matlist
from . import augment as Aug


class StableDiffusion(Dataset):
    def __init__(
        self,
        split,
        pseudo_labels,
        transform,
        renderer,
        matlist,
        use_ref=False,
        dir: typing.Optional[Path] = None,
        **kwargs
    ):
        assert dir.is_dir()
        assert split in ['train', 'valid', 'all']

        self.split = split
        self.renderer = renderer
        self.pseudo_labels = pseudo_labels
        self.use_ref = use_ref
        # self.pl_dir = pl_dir

        if matlist == None:
            files = sorted(dir.rglob('**/outputs/*[0-9].png'))
            files += sorted(dir.rglob('**/out_renorm/*[0-9].png'))
            print(f'total={len(files)}')
            files = [x for x in files if not (x.parent/f'{x.stem}_roughness.png').is_file()]
            print(f'after={len(files)}')
        else:
            files = get_matlist(matlist, dir)

        ### Train/Validation Split
        k = int(len(files)*.98)
        if split == 'train':
            self.files = files[:k]
        elif split == 'valid':
            self.files = files[k:]
        elif split == 'all':
            self.files = files

        random.shuffle(self.files)

        print(f'StableDiffusion list={matlist}:{self.split}=[{len(self.files)}/{len(files)}]')

        dtypes = ['input']
        self.tf = Aug.Pipeline(*transform, dtypes=dtypes)

    def __getitem__(self, index):
        path = self.files[index]
        name = path.stem

        o = EasyDict(dir=str(path.parent), name=name)

        I = tf.to_tensor(Image.open(path).convert('RGB'))

        o.path = str(path)
        o.input, *_ = self.tf([I])
        return o

    def __len__(self):
        return len(self.files)
