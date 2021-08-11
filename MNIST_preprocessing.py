#! /usr/bin/env python3

import glob
import os
import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def scaled(x):
    '''assumes that values of img are between 0 and 255'''
    x = x / 255
    min_, max_ = -1, 1
    x = x * (max_ - min_) + min_
    return x

def main(args):

    # read data
    data = []
    out = ['train', 'valid', 'test']
    for i, file_ in enumerate(['train.csv', 'test.csv']):
        print(i, file_)
        data = pd.read_csv(os.path.join(args.data_dir, file_))
        if i == 0:
            X = np.array(data.iloc[:,1:]).reshape(-1, args.img_size, args.img_size)
            y = np.array(data.iloc[:,0])
            print(f'data read: {data.head()}')
            print(f'data shape {X.shape}')

            # generate train and valid set
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)            

        if i == 1:
            X_test = np.array(data).reshape(-1, args.img_size, args.img_size)
        print(f'labels shape {y.shape}')

    y = [y_train, y_valid]
    for i, X in enumerate([X_train, X_valid, X_test]):
        X = scaled(X)
        # save data
        h5_file = h5py.File(os.path.join(args.data_dir, f'{out[i]}.h5'), 'w')
        chunks = (1,) + X.shape[1:]
        h5_file.create_dataset('X', data=X, shape=X.shape, chunks=chunks)
        if i <= 1:
            h5_file.create_dataset('y', data=y[i], shape=y[i].shape)
        print(f'wrote {out[i]} to h5')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data/MNIST')
    parser.add_argument('-is','--img-size', type=int, default=28)
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f'{key}: {value}')

    main(args)
