import typing
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from . import augment as Aug
from ..render import Renderer
from .utils import MultiLoader, EmptyDataset, collate_fn
from .source import AmbientCG
from .target import StableDiffusion


def is_set(x):
    return x is not None

class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 1,
        transform: bool = None,
        source_ds: str = '',
        target_ds: str = '',
        test_ds: str = '',
        predict_ds: str = '',
        source_list: typing.Optional[Path] = None,
        target_list: typing.Optional[Path] = None,
        target_val_list: typing.Optional[Path] = None,
        test_list: typing.Optional[Path] = None,
        predict_list: typing.Optional[Path] = None,
        source_dir: typing.Optional[Path] = None,
        target_dir: typing.Optional[Path] = None,
        predict_dir: typing.Optional[Path] = None,
        test_dir: typing.Optional[Path] = None,
        pseudo_labels: bool = False,
        input_size: int = 512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.pseudo_labels = pseudo_labels

        self.source_ds = source_ds
        self.target_ds = target_ds
        self.test_ds = test_ds
        self.predict_ds = predict_ds

        self.source_list = source_list
        self.target_list = target_list
        self.target_val_list = target_val_list
        self.predict_list = predict_list
        self.test_list = test_list

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.predict_dir = predict_dir
        self.test_dir = test_dir

        self.input_size = input_size
        # self.use_ref = use_ref

        assert self.source_ds or self.target_ds or self.test_ds or self.predict_ds
        if self.source_ds:
            assert is_set(source_list)
        if self.target_ds:
            assert is_set(target_list)
            if self.target_ds != 'sd':
                assert is_set(target_val_list)

    def setup(self, stage: str):
        renderer = Renderer(return_params=True)
        eval_tf = [
            Aug.NormalizeGeometry(),
            # Aug.CenterCrop((2048,2048)),
            Aug.Resize([self.input_size, self.input_size], antialias=True)]


        if stage == 'fit':
            if self.transform:
                train_tf = [
                    Aug.RandomResizedCrop((512,512), scale=(1/16, 1/4), ratio=(1.,1.)),
                    # Aug.RandomCrop(self.input_size),
                    Aug.NormalizeGeometry(),
                    Aug.RandomHorizontalFlip(),
                    Aug.RandomVerticalFlip(),
                    Aug.RandomIncrementRotate(p=1.),
                    Aug.ColorJitter(brightness=.2, hue=.05, contrast=0.1)
                ]
            else:
                train_tf = [
                    Aug.CenterCrop((self.input_size, self.input_size)),
                    Aug.NormalizeGeometry()]
            train_kwargs = dict(pseudo_labels=self.pseudo_labels,
                                renderer=renderer,
                                transform=train_tf)
            print('stage fit:')
            pprint(train_kwargs)

            ## SOURCE train dataset
            if self.source_ds == 'acg':
                self.src_train = AmbientCG(split='train',
                                           dir=self.source_dir,
                                           matlist=self.source_list,
                                           **train_kwargs)
            ## TARGET train dataset
            if self.target_ds == 'sd':
                self.tgt_train = StableDiffusion(split='train',
                                                #  use_ref=self.use_ref,
                                                 dir=self.target_dir,
                                                 matlist=self.target_list,
                                                 **train_kwargs)

            if not self.source_ds:
                self.src_train = EmptyDataset(len(self.tgt_train))
            if not self.target_ds:
                self.tgt_train = EmptyDataset(len(self.src_train))

        if stage == 'fit' or stage == 'validate':
            validate_kwargs = dict(transform=eval_tf,
                                   renderer=renderer,
                                   set_seed_render=True)

            ## SOURCE validation dataset
            if self.source_ds == 'acg':
                self.src_valid = AmbientCG(split='valid',
                                           dir=self.source_dir,
                                           matlist=self.source_list,
                                           **validate_kwargs)
            ## TARGET validation dataset
            if self.target_ds == 'sd':
                self.tgt_valid = StableDiffusion(split='valid',
                                                 pseudo_labels=False,
                                                 dir=self.target_dir,
                                                #  use_ref=self.use_ref,
                                                 matlist=self.target_list,
                                                 **validate_kwargs)

            if not self.source_ds:
                self.src_valid = EmptyDataset(len(self.tgt_valid))
            if not self.target_ds:
                self.tgt_valid = EmptyDataset(len(self.src_valid))

        elif stage == 'test':
            assert self.test_ds

            test_kwargs = dict(pseudo_labels=False,
                               matlist=self.test_list,
                               transform=eval_tf,
                               renderer=renderer,
                               dir=self.test_dir,
                               set_seed_render=True)

            if self.test_ds == 'acg':
                self.eval = [AmbientCG(split='all', **test_kwargs)]
            elif self.test_ds == 'sd':
                self.eval = [StableDiffusion(split='all', **test_kwargs)]

        elif stage == 'predict':
            predict_kwargs = dict(split='all',
                                  pseudo_labels=False,
                                  dir=self.predict_dir,
                                  matlist=None,
                                  transform=eval_tf,
                                  renderer=renderer)

            if self.predict_ds == 'sd':
                self.ds = StableDiffusion(**predict_kwargs)

    def train_dataloader(self):
        src_dl = DataLoader(dataset=self.src_train,
                            batch_size=self.batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn)
        tgt_dl = DataLoader(dataset=self.tgt_train,
                            batch_size=self.batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn)

        mix = MultiLoader(src_dl, tgt_dl)
        return mix

    def val_dataloader(self):
        src_dl = DataLoader(dataset=self.src_valid,
                            batch_size=self.batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn)
        tgt_dl = DataLoader(dataset=self.tgt_valid,
                            batch_size=self.batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn)

        mix = MultiLoader(src_dl, tgt_dl)
        return mix

    def test_dataloader(self):
        return [DataLoader(dataset=ds,
                           batch_size=self.batch_size,
                           drop_last=False,
                           shuffle=False,
                           num_workers=self.num_workers) for ds in self.eval]

    def predict_dataloader(self):
        return DataLoader(dataset=self.ds,
                          batch_size=1,
                          drop_last=False,
                          shuffle=False,
                          num_workers=1)