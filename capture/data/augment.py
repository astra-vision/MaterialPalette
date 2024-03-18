import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tf
import torch.nn.functional as F


class RandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x, dtypes):
        """WARNING: torchvision v0.11. Wrapper to T.RandomResizedCrop.__call__"""
        i, j, h, w = self.get_params(x[0], self.scale, self.ratio)
        return [tf.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in x]

class ColorJitter(T.ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, dtypes):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        o = []
        for img, dtype in zip(x, dtypes):
            if dtype == 'albedo':
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = tf.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = tf.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = tf.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = tf.adjust_hue(img, hue_factor)
            o.append(img)
        return o

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flip_x(self, img, dtype):
        if dtype == 'normals':
            img[0] *= -1
        return tf.hflip(img)

    def forward(self, x, dtypes):
        if torch.rand(1) < self.p:
            return [self.flip_x(img, dtype) for img, dtype in zip(x, dtypes)]
        return x

class RandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flip_y(self, img, dtype):
        if dtype == 'normals':
            img[1] *= -1
        return tf.vflip(img)

    def forward(self, x, dtypes):
        if torch.rand(1) < self.p:
            return [self.flip_y(img, dtype) for img, dtype in zip(x, dtypes)]
        return x

def deg0(x, y, z):
    return torch.stack([ x,  y,  z])
def deg90(x, y, z):
    return torch.stack([-y,  x,  z])
def deg180(x, y, z):
    return torch.stack([-x, -y,  z])
def deg270(x, y, z):
    return torch.stack([ y, -x,  z])

class RandomIncrementRotate:
    def __init__(self, p):
        self.p = p
        self.angles = [0, 90, 180, 270]

        # adjusts surface normals vector depending on rotation angle
        self.f = { 0: deg0, 90: deg90, 180: deg180, 270: deg270 }

    def rotate(self, img, theta, dtype):
        if dtype == 'normals':
            img = self.f[theta](*img)
        return tf.rotate(img, theta)

    def __call__(self, x, dtypes):
        if torch.rand(1) < self.p:
            theta = random.choice(self.angles)
            return [self.rotate(img, theta, dtype) for img, dtype in zip(x, dtypes)]
        return x

class NormalizeGeometry:
    def normalize(self, img, dtype):
        if dtype == 'normals':
            # perform [0, 1] -> [-1, 1] mapping
            img = 2*img - 1
            # normalize vector to unit sphere
            img = F.normalize(img, dim=0)
        return img

    def __call__(self, x, dtypes):
        return [self.normalize(img, dtype) for img, dtype in zip(x, dtypes)]

class RandomCrop(T.RandomCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, dtypes):
        img_size = tf.get_image_size(x[0])
        assert all(tf.get_image_size(y) == img_size  for y in x)
        i, j, h, w = self.get_params(x[0], self.size)
        return [tf.crop(img, i, j, h, w) for img in x]

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x, dtypes):
        return [tf.center_crop(img, self.size) for img in x]

class Resize(T.Resize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, dtypes):
        return [super(Resize, self).forward(img) for img in x]

class Identity():
    def __call__(self, x, dtypes):
        return x

class ToTensor:
    def __call__(self, x, dtypes):
        return [tf.to_tensor(img) for img in x]

class Pipeline:
    DATA_TYPES = ['input', 'normals', 'albedo']

    def __init__(self, *transforms, dtypes=None):
        assert all(d in Pipeline.DATA_TYPES for d in dtypes)
        self.dtypes = dtypes
        self.transforms = transforms

    def __call__(self, x):
        assert len(self.dtypes) == len(x)
        assert all(y.shape[1:] == x[0].shape[1:] for y in x)
        for f in self.transforms:
            x = f(x, self.dtypes)
        return x