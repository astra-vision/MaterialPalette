
from pathlib import Path

import jsonargparse
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from ..source import Vanilla, DenseReg
from ..callbacks import VisualizeCallback
from ..data.module import DataModule


#! refactor this simplification required
class LightningArgumentParser(jsonargparse.ArgumentParser):
    """
    Extension of jsonargparse.ArgumentParser to parse pl.classes and more.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_datamodule(self, datamodule_obj: pl.LightningDataModule):
        self.add_method_arguments(datamodule_obj, '__init__', 'data', as_group=True)

    def add_lossmodule(self, lossmodule_obj: nn.Module):
        self.add_class(lossmodule_obj, 'loss')

    def add_routine(self, model_obj: pl.LightningModule):
        skip = {'ae', 'decoder', 'loss', 'transnet', 'model', 'discr', 'adv_loss', 'stage'}
        self.add_class_arguments(model_obj, 'routine', as_group=True, skip=skip)

    def add_logger(self, logger_obj):
        skip = {'version', 'config', 'name', 'save_dir'}
        self.add_class_arguments(logger_obj, 'logger', as_group=True, skip=skip)

    def add_class(self, cls, group, **kwargs):
        self.add_class_arguments(cls, group, as_group=True, **kwargs)

    def add_trainer(self):
        skip = {'default_root_dir', 'logger', 'callbacks'}
        self.add_class_arguments(pl.Trainer, 'trainer', as_group=True, skip=skip)

def get_args(datamodule=DataModule, loss=DenseReg, routine=Vanilla, viz=VisualizeCallback):
    parser = LightningArgumentParser()

    parser.add_argument('--config', action=jsonargparse.ActionConfigFile, required=True)
    parser.add_argument('--archi', type=str, required=True)
    parser.add_argument('--out_dir', type=lambda x: Path(x), required=True)

    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--load_weights_from', type=lambda x: Path(x))
    parser.add_argument('--save_ckpt_every', default=10, type=int)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--mode', choices=['train', 'eval', 'test', 'predict'], default='train', type=str)
    parser.add_argument('--resume_from', default=None, type=str)

    if datamodule is not None:
        parser.add_datamodule(datamodule)

    if loss is not None:
        parser.add_lossmodule(loss)

    if routine is not None:
        parser.add_routine(routine)

    if viz is not None:
        parser.add_class_arguments(viz, 'viz', skip={'out_dir', 'exist_ok'})

    # bindings between modules (data/routine/loss)
    parser.link_arguments('data.batch_size', 'routine.batch_size')

    parser.add_logger(WandbLogger)
    parser.add_trainer()

    args = parser.parse_args()
    return args