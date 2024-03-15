from pathlib import Path

import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Resize

from capture.render import encode_as_unit_interval, gamma_encode


class VisualizeCallback(pl.Callback):
    def __init__(self, exist_ok: bool, out_dir: Path, log_every_n_epoch: int, n_batches_shown: int):
        super().__init__()

        self.out_dir = out_dir/'images'
        if not exist_ok and (self.out_dir.is_dir() and len(list(self.out_dir.iterdir())) > 0):
            print(f'directory {out_dir} already exists, press \'y\' to proceed')
            x = input()
            if x != 'y':
                exit(1)

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_epoch = log_every_n_epoch
        self.n_batches_shown = n_batches_shown
        self.resize = Resize(size=[128,128], antialias=True)

    def setup(self, trainer, module, stage):
        self.logger = trainer.logger

    def on_train_batch_end(self, *args):
        self._on_batch_end(*args, split='train')

    def on_validation_batch_end(self, *args):
        self._on_batch_end(*args, split='valid')

    def _on_batch_end(self, trainer, module, outputs, inputs, batch, *args, split):
        x_src, x_tgt = inputs

        # optim_idx:0=discr & optim_idx:1=generator
        y_src, y_tgt = outputs[1]['y'] if isinstance(outputs, list) else outputs['y']

        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epoch == 0 and batch <= self.n_batches_shown:
            if x_src and y_src:
                self._visualize_src(x_src, y_src, split=split, epoch=epoch, batch=batch, ds='src')
            if x_tgt and y_tgt:
                self._visualize_tgt(x_tgt, y_tgt, split=split, epoch=epoch, batch=batch, ds='tgt')

    def _visualize_src(self, x, y, split, epoch, batch, ds):
        zipped = zip(x.albedo, x.roughness, x.normals, x.displacement, x.input, x.image,
                     y.albedo, y.roughness, y.normals, y.displacement, y.reco, y.image)

        grid = [self._visualize_single_src(*z) for z in zipped]

        name = self.out_dir/f'{split}{epoch:05d}_{ds}_{batch}.jpg'
        save_image(grid, name, nrow=1, padding=5)

    @torch.no_grad()
    def _visualize_single_src(self, a, r, n, d, input, mv, a_p, r_p, n_p, d_p, reco, mv_p):
        n = encode_as_unit_interval(n)
        n_p = encode_as_unit_interval(n_p)

        mv_gt = [gamma_encode(o) for o in mv]
        mv_pred = [gamma_encode(o) for o in mv_p]
        reco = gamma_encode(reco)

        maps = [input, a, r, n, d] + mv_gt + [reco, a_p, r_p, n_p, d_p] + mv_pred
        maps = [self.resize(x.cpu()) for x in maps]
        return make_grid(maps, nrow=len(maps)//2, padding=0)

    def _visualize_tgt(self, x, y, split, epoch, batch, ds):
        zipped = zip(x.input, y.albedo, y.roughness, y.normals, y.displacement)

        grid = [self._visualize_single_tgt(*z) for z in zipped]

        name = self.out_dir/f'{split}{epoch:05d}_{ds}_{batch}.jpg'
        save_image(grid, name, nrow=1, padding=5)

    @torch.no_grad()
    def _visualize_single_tgt(self, input, a_p, r_p, n_p, d_p):
        n_p = encode_as_unit_interval(n_p)
        maps = [input, a_p, r_p, n_p, d_p]
        maps = [self.resize(x.cpu()) for x in maps]
        return make_grid(maps, nrow=len(maps), padding=0)