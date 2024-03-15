from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from ..render import Renderer, Scene, generate_random_scenes, generate_specular_scenes


class RenderingLoss(nn.Module):
    def __init__(self, renderer, n_random_configs=0, n_symmetric_configs=0):
        super().__init__()
        self.eps = 0.1
        self.renderer = renderer
        self.n_random_configs = n_random_configs
        self.n_symmetric_configs = n_symmetric_configs
        self.n_renders = n_random_configs + n_symmetric_configs

    def generate_scenes(self):
        return generate_random_scenes(self.n_random_configs) + generate_specular_scenes(self.n_symmetric_configs)

    def multiview_render(self, y, x):
        X_renders, Y_renders = [], []

        x_svBRDFs = zip(x.normals, x.albedo, x.roughness, x.displacement)
        y_svBRDFs = zip(y.normals, y.albedo, y.roughness, x.displacement)
        for x_svBRDF, y_svBRDF in zip(x_svBRDFs, y_svBRDFs):
            x_renders, y_renders = [], []
            for scene in self.generate_scenes():
                x_renders.append(self.renderer.render(scene, x_svBRDF))
                y_renders.append(self.renderer.render(scene, y_svBRDF))
            X_renders.append(torch.cat(x_renders))
            Y_renders.append(torch.cat(y_renders))

        out = torch.stack(X_renders), torch.stack(Y_renders)
        return out

    def reconstruction(self, y, theta):
        views = []
        for *svBRDF, t in zip(y.normals, y.albedo, y.roughness, y.displacement, theta):
            render = self.renderer.render(Scene.load(t), svBRDF)
            views.append(render)
        return torch.cat(views)

    def __call__(self, y, x, **kargs):
        loss = F.l1_loss(torch.log(y + self.eps), torch.log(x + self.eps), **kargs)
        return loss

class DenseReg(nn.Module):
    def __init__(
        self,
        reg_weight: float,
        render_weight: float,
        pl_reg_weight: float = 0.,
        pl_render_weight: float = 0.,
        use_source: bool = True,
        use_target: bool = True,
        n_random_configs= 3,
        n_symmetric_configs = 6,
    ):
        super().__init__()

        self.weights = [('albedo', reg_weight, self.log_l1),
                        ('roughness', reg_weight, self.log_l1),
                        ('normals', reg_weight, F.l1_loss)]

        self.reg_weight = reg_weight
        self.render_weight = render_weight
        self.pl_reg_weight = pl_reg_weight
        self.pl_render_weight = pl_render_weight
        self.use_source = use_source
        self.use_target = use_target

        self.renderer = Renderer()
        self.n_random_configs = n_random_configs
        self.n_symmetric_configs = n_symmetric_configs
        self.loss = RenderingLoss(self.renderer, n_random_configs=n_random_configs, n_symmetric_configs=n_symmetric_configs)

    def log_l1(self, x, y, **kwargs):
        return F.l1_loss(torch.log(x + 0.01), torch.log(y + 0.01), **kwargs)

    def forward(self, x, y):
        loss = EasyDict()
        x_src, x_tgt = x
        y_src, y_tgt = y

        if self.use_source:
            # acg regression loss
            for k, w, loss_fn in self.weights:
                loss[k] = w*loss_fn(y_src[k], x_src[k])

            # rendering loss
            x_src.image, y_src.image = self.loss.multiview_render(y_src, x_src)
            loss.render = self.render_weight*self.loss(y_src.image, x_src.image)

            # reconstruction
            y_src.reco = self.loss.reconstruction(y_src, x_src.input_params)

        if self.use_target:
            for k, w, loss_fn in self.weights:
                loss[f'tgt_{k}'] = self.pl_reg_weight*loss_fn(y_tgt[k], x_tgt[k])

            # rendering loss w/ pseudo label
            y_tgt.image, x_tgt.image = self.loss.multiview_render(y_tgt, x_tgt)
            loss.sd_render = self.pl_render_weight*self.loss(y_tgt.image, x_tgt.image)

            # reconstruction
            y_tgt.reco = self.loss.reconstruction(y_tgt, x_tgt.input_params)

        loss.total = torch.stack(list(loss.values())).sum()
        return loss

    @torch.no_grad()
    def test(self, x, y, batch_idx, epoch, dl_id):
        assert len(x.name) == 1
        y.reco = self.loss.reconstruction(y, x.input_params)
        return EasyDict(total=0)

    @torch.no_grad()
    def predict(self, x_tgt, y_tgt, batch_idx, split, epoch):
        assert len(x_tgt.name) == 1

        # gt components
        I = x_tgt.input[0]
        name = x_tgt.name[0]

        # get the predicted maps
        N_pred = y_tgt.normals[0]
        A_pred = y_tgt.albedo[0]
        R_pred = y_tgt.roughness[0]

        # A_name = pl_path/f'{name}_albedo.png'
        # save_image(A_pred, A_name)

        # N_name = pl_path/f'{name}_normals.png'
        # save_image(encode_as_unit_interval(N_pred), N_name)

        # R_name = pl_path/f'{name}_roughness.png'
        # save_image(R_pred, R_name)

        return EasyDict(total=0)
