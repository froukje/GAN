#! /usr/bin/env python3

import glob
import os
import argparse
import h5py
import numpy as np
from PIL import Image

def scaled(x):
    '''assumes that values of img are between 0 and 1'''
    min_, max_ = -1, 1
    x = x * (max_ - min_) + min_
    return x

def main(args):

    # read data
    data = []
    out = ['train', 'valid']
    for i, folder in enumerate(args.folders):
        print(i, folder)
        path = args.data_dir + folder #os.path.join(args.data_dir, folder)
        for image in glob.glob(os.path.join(path,'*.png')):
            img = Image.open(image)
            img = img.resize((args.img_size, args.img_size))
            img = np.array(img) / 255 # normalize to values between 0 and 1
            # normalize between -1 and +1
            # we will use tanh activation funtion in generator, which will contain pixel values from -1 to 1
            img = scaled(img)
            data.append(img)
        data_array = np.stack(data)
        print(f'folder data shape: {data_array.shape}')

        # save data
        h5_file = h5py.File(os.path.join(args.data_dir, f'{out[i]}.h5'), 'w')
        chunks = (1,) + data_array.shape[1:]
        h5_file.create_dataset('X', data=data_array, shape=data_array.shape, chunks=chunks)
        print(f'wrote {out[i]} to h5')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data/SVHN')
    parser.add_argument('-f', '--folders', type=str, nargs='+', default=['/train/train/', 'test/test/'])
    parser.add_argument('-is','--img-size', type=int, default=64)
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f'{key}: {value}')

    main(args)
