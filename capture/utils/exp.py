import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer as plTrainer
from pathlib import Path

from ..callbacks import VisualizeCallback, MetricLogging
from ..data.module import DataModule
from .log import get_info


def get_data(args=None, **kwargs):
    if args is None:
        return DataModule(**kwargs)
    else:
        return DataModule(**args.data)

def get_name(args) -> str:
    name = ''#f'{args.mode}'

    src_ds_verbose = str(args.data.source_list).split(os.sep)[-1].replace("_","-").upper()

    if args.mode == 'train':
        if args.loss.use_source and not args.loss.use_target:
            name += f'pretrain_ds{args.data.source_ds.upper()}_lr{args.routine.lr}_x{args.data.input_size}_bs{args.data.batch_size}_reg{args.loss.reg_weight}_rend{args.loss.render_weight}_ds{str(args.data.source_list).split(os.sep)[-1].replace("_","-").upper()}'

        elif args.loss.use_target:
            name += f'_F_{args.data.target_ds.upper()}_lr{args.routine.lr}_x{args.data.input_size}_bs{args.data.tgt_bs}_aug{int(args.data.transform)}_reg{args.loss.pl_reg_weight}_rend{args.loss.pl_render_weight}_ds{str(args.data.target_list).split(os.sep)[-1].replace("_","-").upper()}'
        else:
            name += f'_T_{args.data.source_ds.upper()}_lr{args.routine.lr}_x{args.data.input_size}_aug{int(args.data.transform)}_reg{args.loss.render_weight}_rend{args.loss.render_weight}_ds{str(args.data.source_list).split(os.sep)[-1].replace("_","-").upper()}'

        if args.loss.adv_weight:
            name += f'_ADV{args.loss.adv_weight}'
        if args.data.source_ds == 'acg':
            name += f'_mixbs{args.data.batch_size}'
        if args.loss.reg_weight != 0.1:
            name += f'_regSRC{args.loss.reg_weight}'
        if args.loss.render_weight != 1:
            name += f'_rendSRC{args.loss.render_weight}'
        if args.data.use_ref:
            name += '_useRef'
        if args.load_weights_from:
            wname, epoch = get_info(str(args.load_weights_from))
            assert wname and epoch
            name += f'_init{wname.replace("_", "-")}-{epoch}ep'

    name += f'_s{args.seed}'
    return name
    # name += args.load_weights_from.split(os.sep)[-1][:-5]

def get_callbacks(args):
    callbacks = [
        VisualizeCallback(out_dir=args.out_dir, exist_ok=bool(args.resume_from), **args.viz),
        ModelCheckpoint(
            dirpath=args.out_dir/'ckpt',
            filename='{name}_{epoch}-{step}',
            save_weights_only=False,
            save_top_k=-1,
            every_n_epochs=args.save_ckpt_every),
        MetricLogging(args.load_weights_from, args.data.test_list, outdir=Path('./logs')),
    ]
    return callbacks

class Trainer(plTrainer):
    def __init__(self, o_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_path = o_args.resume_from

    def __call__(self, mode, module, data) -> None:
        if mode == 'test':
            self.test(module, data)
        elif mode == 'eval':
            self.validate(module, data)
        elif mode == 'predict':
            self.predict(module, data)
        elif mode == 'train':
            self.fit(module, data, ckpt_path=self.ckpt_path)