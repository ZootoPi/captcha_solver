from PIL import Image
import numpy as np

from model import AlexNet

model = AlexNet()

print(model)

im = Image.open(
    'images/char-4-epoch-6/train/0123_9ecd041b-b929-4f01-afdd-b7b2e73a7830.png')

# im = preprocess(im)
# input_batch = im.unsqueeze(0)

y = model(im)

print(y)
