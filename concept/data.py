from pathlib import Path
import math
from PIL import Image

import torch
import torchvision.transforms.functional as tf
import torchvision.transforms as T
from torchvision.transforms import RandomRotation
import torch.utils.checkpoint
from torch.utils.data import Dataset


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, data_dir, prompt, tokenizer, size=512):
        super().__init__()
        self.size = size
        self.tokenizer = tokenizer

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = [x for x in data_dir.iterdir() if x.is_file()]
        assert len(self) > 0, 'data directory is empty'

        self.prompt = prompt

        self.image_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        image = Image.open(self.instance_images_path[index % len(self)])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = self.image_transforms(image)
        prompt = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return img, prompt