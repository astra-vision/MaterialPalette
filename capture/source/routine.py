import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from easydict import EasyDict
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure

from . import DenseReg, RenderingLoss
from ..render import Renderer, encode_as_unit_interval, gamma_decode, gamma_encode


class Vanilla(pl.LightningModule):
    metrics = ['I_mse','N_mse','A_mse','R_mse','I_ssim','N_ssim','A_ssim','R_ssim']
    maps = {'I': 'reco', 'N': 'normals', 'R': 'roughness', 'A': 'albedo'}

    def __init__(self, model: nn.Module, loss: DenseReg = None, lr: float = 0, batch_size: int = 0):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.tanh = nn.Tanh()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def training_step(self, x):
        y = self(*x)
        loss = self.loss(x, y)
        self.log_to('train', loss)
        return dict(loss=loss.total, y=y)

    def forward(self, src, tgt):
        src_out, tgt_out = None, None

        if None not in src:
            src_out = self.model(self.norm(src.input))
            self.post_process_(src_out)

        if None not in tgt:
            tgt_out = self.model(self.norm(tgt.input))
            self.post_process_(tgt_out)

        return src_out, tgt_out

    def post_process_(self, o: EasyDict):
        # (1) activation function, (2) concat unit z, (3) normalize to unit vector
        nxy = self.tanh(o.normals)
        nx, ny = torch.split(nxy*3, split_size_or_sections=1, dim=1)
        n = torch.cat([nx, ny, torch.ones_like(nx)], dim=1)
        o.normals = F.normalize(n, dim=1)

        # (1) activation function, (2) mapping [-1,1]->[0,1]
        a = self.tanh(o.albedo)
        o.albedo = encode_as_unit_interval(a)

        # (1) activation function, (2) mapping [-1,1]->[0,1], (3) channel repeat x3
        r = self.tanh(o.roughness)
        o.roughness = encode_as_unit_interval(r.repeat(1,3,1,1))

    def validation_step(self, x, *_):
        y = self(*x)
        loss = self.loss(x, y)
        self.log_to('val', loss)
        return dict(loss=loss.total, y=y)

    def log_to(self, split, loss):
        self.log_dict({f'{split}/{k}': v for k, v in loss.items()}, batch_size=self.batch_size)

    def on_test_start(self):
        self.renderer = RenderingLoss(Renderer())

        for m in Vanilla.metrics:
            if 'mse' in m:
                setattr(self, m, MeanSquaredError().to(self.device))
            elif 'ssim' in m:
                setattr(self, m, StructuralSimilarityIndexMeasure(data_range=1).to(self.device))

    def test_step(self, x, batch_idx, dl_id=0):
        y = self.model(self.norm(x.input))
        self.post_process_(y)

        # image reconstruction
        y.reco = self.renderer.reconstruction(y, x.input_params)
        x.reco = gamma_decode(x.input)

        for m in Vanilla.metrics:
            mapid, *_ = m
            k = Vanilla.maps[mapid]
            meter = getattr(self, m)
            meter(y[k], x[k].to(y[k].dtype))
            self.log(m, getattr(self, m), on_epoch=True)

    def predict_step(self, x, batch_idx):
        y = self.model(self.norm(x.input))
        self.post_process_(y)

        I, name, outdir = x.input[0], x.name[0], Path(x.path[0]).parent
        N_pred, A_pred, R_pred  = y.normals[0], y.albedo[0], y.roughness[0]

        save_image(gamma_encode(A_pred), outdir/f'{name}_albedo.png')
        save_image(encode_as_unit_interval(N_pred), outdir/f'{name}_normals.png')
        save_image(R_pred, outdir/f'{name}_roughness.png')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return dict(optimizer=optimizer)