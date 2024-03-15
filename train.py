from pytorch_lightning import seed_everything
from capture.utils import Trainer, get_args, get_module, get_name, get_data


if __name__ == '__main__':
    args = get_args()

    seed_everything(seed=args.seed)

    data = get_data(args.data)
    module = get_module(args)

    args.name = get_name(args)
    args.out_dir = args.out_dir/name
    callbacks = get_callbacks(args)
    logger = get_logger(args)

    trainer = Trainer(args, default_root_dir=out_dir, logger=logger, callbacks=callbacks, **args.trainer)
    trainer(args.mode, module, data)