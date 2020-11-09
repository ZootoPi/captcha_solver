'''
This module will generate captcha image for train, val and test phase
Usage: gen_data.py [-h] [-n N] [-t T] [-d] [-l] [-u] [--npi NPI] [--data_dir DATA_DIR]

optional arguments:
  -h, --help           show this help message and exit
  -n N                 epoch number of character permutations.
  -t T                 ratio of test dataset.
  -d, --digit          use digits in dataset.
  -l, --lower          use lowercase in dataset.
  -u, --upper          use uppercase in dataset.
  --npi NPI            number of characters per image.
  --data_dir DATA_DIR  where data will be saved.
'''

import argparse
import json
import itertools
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha


FLAGS = None
META_FILENAME = 'meta.json'


def _get_choices():
    choices = [
        (FLAGS.digit, "0123456789"),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
    ]
    return tuple([i for is_selected, subset in choices for i in subset if is_selected])


def _gen_captcha(img_dir, num_per_image, n_epoch, width, height, choices):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    print('generating %s epochs of captcha in %s' % (n_epoch, img_dir))
    for _ in range(n_epoch):
        for i in itertools.permutations(choices, num_per_image):
            captcha = ''.join(i)
            file_path = os.path.join(img_dir,  '%s_%s.png' %
                                     (captcha, uuid.uuid4()))
            image.write(captcha, file_path)


def _build_file_path(file_name):
    return os.path.join(FLAGS.data_dir, 'char-%s-epoch-%s' % (FLAGS.npi, FLAGS.n), file_name)


def _gen_dataset():
    n_epoch = FLAGS.n
    num_per_image = FLAGS.npi
    test_ratio = FLAGS.t

    choices = _get_choices()

    width = 40 + 20 * num_per_image
    height = 100

    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(choices),
        'label_choices': ''.join(choices),
        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }

    print('%s choices: %s' % (len(choices), ''.join(choices) or None))

    _gen_captcha(_build_file_path('train'), num_per_image,
                 n_epoch, width, height, choices=choices)
    _gen_captcha(_build_file_path('val'), num_per_image, max(
        1, int(n_epoch * test_ratio)), width, height, choices=choices)
    _gen_captcha(_build_file_path('test'), num_per_image, max(
        1, int(n_epoch * test_ratio)), width, height, choices=choices)

    meta_filename = _build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as file:
        json.dump(meta, file, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        default=6,
        type=int,
        help='epoch number of character permutations.')

    parser.add_argument(
        '-t',
        default=0.2,
        type=float,
        help='ratio of test dataset.')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=4,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')

    FLAGS, unparsed = parser.parse_known_args()

    _gen_dataset()
