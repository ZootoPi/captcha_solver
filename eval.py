'''
Evaluate the model

usage: eval.py [-h] [-m MODEL] [-d DATA_DIR] [--batch BATCH]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to trained model.
  -d DATA_DIR, --data_dir DATA_DIR
                        image data folder.
  --batch BATCH         batch size.
'''

import argparse
import torch
from torchvision import transforms

from utils import load_data


def create_dataloader(data_dir, batch_size):
    '''
    Create dataloader
    '''
    im_transforms = transforms.Compose([
        transforms.Resize((120, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    _, image_datasets = load_data(
        data_dir, transform=im_transforms)

    return torch.utils.data.DataLoader(
        image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=4)


def evaluate(model_path, data_dir, batch_size):
    '''
    Evaluate the model
    '''
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create dataloader
    dataloader = create_dataloader(data_dir, batch_size)

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)

        for x, y in zip(predict, labels.data):
            if torch.equal(x, y):
                running_corrects += 1
            else:
                print(x, y)

    data_len = len(dataloader.dataset)
    print("Test accuracy: ", float(running_corrects) / data_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        default='model.pt',
        type=str,
        help='path to trained model.')

    parser.add_argument(
        '-d', '--data_dir',
        default='images/char-4-epoch-6',
        type=str,
        help='image data folder.')

    parser.add_argument(
        '--batch',
        default=16,
        type=int,
        help='batch size.')

    hp = parser.parse_args()

    evaluate(hp.model, hp.data_dir, hp.batch)
