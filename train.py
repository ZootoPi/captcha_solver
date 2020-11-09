'''
Train model

usage: train.py [-h] [-d DATA_DIR] [--lr LR] [--batch BATCH] [-e EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        image data folder.
  --lr LR               learning rate.
  --batch BATCH         batch size.
  -e EPOCH, --epoch EPOCH
                        number of epoch for training.
'''

import argparse
import copy
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn

from model import AlexNet
from utils import load_data


def feed_data(model, phase, dataloaders, criterion, optimizer, device):
    '''
    Feed data to model and calculate loss, accuracy
    '''

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += sum([torch.equal(x, y)
                                 for x, y in zip(preds, labels.data)])

    data_len = len(dataloaders[phase].dataset)
    return running_loss / data_len, float(running_corrects) / data_len


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    '''
    Train model
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Calculate loss and accuracy
            epoch_loss, epoch_acc = feed_data(
                model, phase, dataloaders, criterion, optimizer, device)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_dataloader(data_dir, batch_size):
    '''
    Create dataloader
    '''
    im_transforms = transforms.Compose([
        transforms.Resize((120, 100)),
        transforms.CenterCrop((120, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    _, image_datasets = load_data(
        data_dir, transform=im_transforms)

    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']}
    return dataloaders_dict


def _main(data_dir, batch_size, learning_rate, n_epoch):
    '''
    main function
    '''
    # Create dataloader
    dataloaders_dict = create_dataloader(data_dir, batch_size)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model
    model = AlexNet()
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model = train_model(
        model, dataloaders_dict, criterion, optimizer_ft, device, n_epoch)

    torch.save(model, 'model.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir',
        default='images/char-4-epoch-6',
        type=str,
        help='image data folder.')

    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='learning rate.')

    parser.add_argument(
        '--batch',
        default=16,
        type=int,
        help='batch size.')
    parser.add_argument(
        '-e', '--epoch',
        default=16,
        type=int,
        help='number of epoch for training.')

    hp = parser.parse_args()

    _main(hp.data_dir, hp.batch, hp.lr, hp.epoch)
