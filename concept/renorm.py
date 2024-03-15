import argparse, csv, random
from pathlib import Path
from PIL import Image

import numpy as np
import cv2
import torch
from tqdm import tqdm
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def renorm(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renorm_dir = path.parent.parent/'out_renorm'
    proj_dir = Path(*path.parts[:-6])

    ## Get input rgb image
    proposals = [x for x in proj_dir.iterdir() if x.is_file() and x.suffix in ('.jpg', '.png', '.jpeg')]
    assert len(proposals) == 1
    img_path = proposals[0]
    pil_img = Image.open(img_path)
    color = tf.to_tensor(pil_img.convert('RGB'))

    ## Get region mask
    mask_path = proj_dir/'masks'/f'{path.parts[-5]}.png'
    assert mask_path.is_file()
    mask = tf.to_tensor(Image.open(mask_path).convert('L'))
    mask = tf.resize(mask, size=(pil_img.height, pil_img.width))[0]

    mask = mask == 1.
    grayscale = tf.to_tensor(pil_img.convert('L'))[0]
    gray_flat = grayscale[mask]

    # Flatten the grayscale and sort pixels
    sorted_pixels, _ = gray_flat.sort()
    exclude_count = int(0.005 * len(gray_flat))
    low_threshold = sorted_pixels[exclude_count]
    high_threshold = sorted_pixels[-exclude_count]

    # construct the mask
    m = (gray_flat >= low_threshold) & (gray_flat <= high_threshold)

    ref_flatten = color[:,mask]
    ref = torch.stack([ref_flatten[0, m], ref_flatten[1, m], ref_flatten[2, m]])
    mean_ref = ref.mean(1)[:,None,None].to(device)
    std_ref = ref.std(1)[:,None,None].to(device)

    # gather patches
    renorm_dir.mkdir(exist_ok=True)
    x = tf.to_tensor(Image.open(path))[None].to(device)
    mean = x.mean(dim=(2,3),keepdim=True)
    std = x.std(dim=(2,3),keepdim=True)

    # renormalize
    x = (x-mean)/std * std_ref + mean_ref
    x.clamp_(0,1)

    s_out = renorm_dir/path.name
    tf.to_pil_image(x[0]).save(s_out)