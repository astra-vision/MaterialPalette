import os
from pathlib import Path
from collections import OrderedDict

from pytorch_lightning.callbacks import Callback

from ..utils.log import append_csv, get_info


class MetricLogging(Callback):
    def __init__(self, weights: str, test_list: str, outdir: Path):
        super().__init__()
        assert outdir.is_dir()

        self.weights = weights
        self.test_list = test_list
        self.outpath = outdir/'eval.csv'

    def on_test_end(self, trainer, pl_module):
        weight_name, epoch = get_info(str(self.weights))
        *_, test_set = self.test_list.parts

        parsed = {k: f'{v}' for k,v in trainer.logged_metrics.items()}

        odict = OrderedDict(name=weight_name, epoch=epoch, test_set=test_set)
        odict.update(parsed)
        append_csv(self.outpath, odict)
        print(f'logged metrics in: {self.outpath}')