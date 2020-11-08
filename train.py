'''
Train model
'''

import time
import copy
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn

from model import AlexNet
from utils import load_data


DATA_DIR = "images/char-4-epoch-6"
BATCH_SIZE = 16


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    '''
    Train model
    '''
    since = time.time()

    val_acc_history = []

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

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, batch in enumerate(dataloaders[phase]):
                inputs, labels = batch
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
            epoch_loss = running_loss / data_len
            epoch_acc = float(running_corrects) / data_len

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Create dataloader
im_transforms = transforms.Compose([
    transforms.Resize((120, 100)),
    transforms.CenterCrop((120, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _main():
    """
    docstring
    """
    _, image_datasets = load_data(
        DATA_DIR, transform=im_transforms)

    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    # Send the model to GPU
    model = model.to(device)

    # Observe that all parameters are being optimized
    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model, _ = train_model(
        model, dataloaders_dict, criterion, optimizer_ft, device, 30)

    torch.save(model, 'model.pt')


if __name__ == "__main__":
    _main()
