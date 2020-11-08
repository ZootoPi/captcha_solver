'''
This file define models for resolve captcha problem
'''

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    '''
    This model is based on Alexnet
    '''

    def __init__(self, num_classes: int = 10, num_digits: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_digits = num_digits

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(9856, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes * num_digits),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Feed-forward
        '''
        tensor = self.features(tensor)
        tensor = torch.flatten(tensor, 1)
        tensor = self.classifier(tensor)
        tensor = torch.reshape(
            tensor, (tensor.shape[0], self.num_classes, self.num_digits))
        return tensor
