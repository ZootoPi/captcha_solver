import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


def load_data(data_dir, flatten=False, transforms=None):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as f:
        meta = json.load(f)

    return (
        meta,
        {
            'train': CaptchaDataset(train_dir, transforms, **meta),
            'val': CaptchaDataset(val_dir, transforms, **meta),
            'test': CaptchaDataset(test_dir, transforms, **meta)
        }
    )


class CaptchaDataset(Dataset):
    """Provide `next_batch` method, which returns the next `batch_size` examples from this data set."""

    def __init__(self, dir, transforms, **meta):
        self.meta = meta
        self.data = self._scan_images(dir, **meta)
        self.transforms = transforms

    def _read_image(self, filename, width, height, **extra_meta):
        im = Image.open(filename).resize(
            (width, height), Image.ANTIALIAS)

        return im

    def _read_label(self, filename, label_choices, **extra_meta):
        basename = os.path.basename(filename)
        labels = basename.split('_')[0]

        data = []

        for c in labels:
            idx = label_choices.index(c)
            data.append(idx)

        return data

    def _scan_images(self, dir, flatten=False, ext='.png', **meta):
        data = []
        for fn in os.listdir(dir):
            if fn.endswith(ext):
                tmp = {}
                fd = os.path.join(dir, fn)
                tmp['image'] = fd
                tmp['labels'] = self._read_label(fd, **meta)
                data.append(tmp)
        return data

    def __getitem__(self, idx):
        d = self.data[idx]
        image = self._read_image(d['image'], **self.meta)

        label = torch.tensor(d['labels'], dtype=torch.long)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.data)
