import typing
from pathlib import Path
from PIL import Image
import cv2

import torch
from easydict import EasyDict
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset

from ..utils.log import get_matlist
from . import augment as Aug


class AmbientCG(Dataset):
    def __init__(
        self,
        split,
        transform,
        renderer,
        matlist,
        dir: typing.Optional[Path] = None,
        set_seed_render: bool = False,
        **kwargs
    ):
        assert dir.is_dir()
        assert matlist.is_file()
        assert split in ['train', 'valid', 'all']
        self.set_seed_render = set_seed_render

        files = get_matlist(matlist, dir)

        # train/val/ split
        self.split = split
        k = int(len(files)*.95)
        if split == 'train':
            self.files = files[:k]
        elif split == 'valid':
            self.files = files[k:]
        elif split == 'all':
            self.files = files

        print(f'AmbientCG list={matlist}:{self.split}=[{len(self.files)}/{len(files)}]')

        dtypes = ['normals', 'albedo', 'input', 'input']
        self.tf = Aug.Pipeline(*transform, dtypes=dtypes)
        self.renderer = renderer

    def __getitem__(self, index, quick=False):
        path = self.files[index]
        name = path.stem.split('_')[0]
        root = path.parent

        N_path = root / f'{name}_2K-PNG_NormalGL.png'
        N = tf.to_tensor(Image.open(N_path).convert('RGB'))

        A_path = root / f'{name}_2K-PNG_Color.png'
        A = tf.to_tensor(Image.open(A_path).convert('RGB'))

        R_path = root / f'{name}_2K-PNG_Roughness.png'
        R = tf.to_tensor(Image.open(R_path).convert('RGB'))

        D_path = root / f'{name}_2K-PNG_Displacement.png'
        D_pil = cv2.imread(str(D_path), cv2.IMREAD_GRAYSCALE)
        D = torch.from_numpy(D_pil)[None].repeat(3,1,1)/255

        # augmentation
        N, A, R, D = self.tf([N, A, R, D])

        if self.set_seed_render:
            torch.manual_seed(hash(name))
        I, params = self.renderer([N, A, R, D], n_samples=1)
        params = torch.stack(params)

        # return homogenous object whatever the source: acg or sd
        return EasyDict(
            input=I[0],
            input_params=params[:,0],
            normals=N,
            albedo=A,
            roughness=R,
            displacement=D,
            name=name,
        )

    def __len__(self):
        return len(self.files)