from pathlib import Path
from argparse import ArgumentParser

from pytorch_lightning import Trainer

import concept
import capture


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()

    ## Extract square crops from image for each of the binary masks located in <path>/masks
    regions = concept.crop(args.path)

    ## Iterate through regions to invert the concept and generate texture views
    for region in regions.iterdir():
        lora = concept.invert(region)
        concept.infer(lora, renorm=True)

    ## Construct a dataset with all generations and load pretrained decomposition model
    data = capture.get_data(predict_dir=args.path, predict_ds='sd')
    module = capture.get_inference_module(pt='model.ckpt')

    ## Proceed with inference on decomposition model
    decomp = Trainer(default_root_dir=args.path, accelerator='gpu', devices=1, precision=16)
    decomp.predict(module, data)