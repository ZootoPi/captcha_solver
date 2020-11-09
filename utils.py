'''
Utils module
'''

import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch


def load_data(data_dir, transform=None):
    '''
    return train, val, test dataset
    '''
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as file:
        meta = json.load(file)

    return (
        meta,
        {
            'train': CaptchaDataset(train_dir, transform, **meta),
            'val': CaptchaDataset(val_dir, transform, **meta),
            'test': CaptchaDataset(test_dir, transform, **meta)
        }
    )


class CaptchaDataset(Dataset):
    '''
    Prepare captcha dataset
    '''

    def __init__(self, folder, transform, **meta):
        self.meta = meta
        self.width = meta['width']
        self.height = meta['height']
        self.label_choices = meta['label_choices']
        self.data = self._scan_images(folder)
        self.transform = transform

    def _read_image(self, filename):
        img = Image.open(filename).resize(
            (self.width, self.height), Image.ANTIALIAS)

        return img

    def _read_label(self, filename):
        basename = os.path.basename(filename)
        labels = basename.split('_')[0]

        data = []

        for char in labels:
            idx = self.label_choices.index(char)
            data.append(idx)

        return data

    def _scan_images(self, folder, ext='.png'):
        data = []
        for file_name in os.listdir(folder):
            if file_name.endswith(ext):
                tmp = {}
                file_path = os.path.join(folder, file_name)
                tmp['image'] = file_path
                tmp['labels'] = self._read_label(file_path)
                data.append(tmp)
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        image = self._read_image(data['image'])

        label = torch.tensor(data['labels'], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
