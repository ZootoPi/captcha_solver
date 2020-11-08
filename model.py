import torch
import torch.nn as nn
from torchvision import transforms


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10, num_digits: int = 4) -> None:
        super(AlexNet, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.reshape(x, (x.shape[0], self.num_classes, self.num_digits))
        return x
