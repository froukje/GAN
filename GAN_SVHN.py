#! /usr/bin/env python3

import glob
import os
import argparse
import h5py
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer

# Dataset
class SVHNDataset(Dataset):
    def __init__(self, data_path, transform=None):
        '''
        data: images
        transform: optional transform to be applied on a sample
        '''
        print(f'load images ({data_path}) ...')
        start_time = time.time()
        h5_file = h5py.File(data_path, 'r')
        self.transform = transform
        self.X = h5_file['X'][:]
        
        print(f'loading images ({data_path}) took {time.time() - start_time:.2f}s')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
        return x
       

# Data Module
class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        
        self.transform = transforms.Compose([transforms.ToTensor()])
                                  
    def setup(self, stage):
        if stage=='fit' or stage is None:
            svhn_train = os.path.join(args.data_dir, 'train.h5')
            svhn_valid = os.path.join(args.data_dir, 'valid.h5')
            
            self.train_dataset = SVHNDataset(svhn_train, transform=self.transform)
            self.valid_dataset = SVHNDataset(svhn_valid, transform=self.transform)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f'train dataloader: {next(iter(train_dataloader))[0].shape}')
        return train_dataloader

    def val_dataloader(self):
        val_dataloader =  DataLoader(self.valid_dataset, batch_size=args.batch_size, drop_last=True)
        print(f'val dataloader: {next(iter(val_dataloader))[0].shape}')
        return val_dataloader

# Discriminator
class SVHNDiscriminator(nn.Module):
    '''Inputs of discriminator are 64x64x3 tensor images'''
    def __init__(self, args):
        super().__init__()

        def disc_block(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 
                        kernel_size, stride, padding=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            layers.append(nn.Dropout(args.disc_dropout))
            return layers

        self.disc_model = nn.Sequential(*disc_block(3, args.conv_dim, 4, batch_norm=False), # 64x64 in, 32x32 out, no batch norm
                                        *disc_block(args.conv_dim, args.conv_dim*2, 4), # 16x16 out
                                        *disc_block(args.conv_dim*2, args.conv_dim*4, 4), # 8x8 out
                                        *disc_block(args.conv_dim*4, args.conv_dim*8, 4), # 4x4 out
                                        )

        self.fc = nn.Linear(args.conv_dim*4*4*2, 1) # final fully connected layer

    def forward(self, img):
        '''flattens input image and performs forward pass'''
        output = self.disc_model(img)
        output = output.view(-1, args.conv_dim*4*4*2)
        output = self.fc(output)
        return output

# Generator
class SVHNGenerator(nn.Module):
    '''input of the generator is a noise vector t,
       the output will be a tanh output with size 64x64 (as the images)'''
    def __init__(self, args):
        super().__init__()

        def gen_block(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, relu=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if relu:
                layers.append(nn.ReLU())
            return layers

        self.fc = nn.Linear(args.z_size, args.conv_dim*4*4*4*2)
        self.gen_model = nn.Sequential(*gen_block(args.conv_dim*8, args.conv_dim*4, 4),
                                       *gen_block(args.conv_dim*4, args.conv_dim*2, 4),
                                       *gen_block(args.conv_dim*2, args.conv_dim, 4),
                                       *gen_block(args.conv_dim, 3, 4, batch_norm=False, relu=False),
                                       nn.Tanh()
                                      )
    def forward(self, z):
        # fully connected + reshape
        z = self.fc(z)
        z = z.view(-1, args.conv_dim*8, 4, 4) # (batch_size, depth, 4, 4)
        z = self.gen_model(z)
        return z
# GAN
class SVHNGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.generator = SVHNGenerator(args)
        self.discriminator = SVHNDiscriminator(args)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        ad_loss = criterion(y_hat, y)
        return ad_loss

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        self.imgs = train_batch

        # sample noise
        z = np.random.uniform(-1, 1, size=(args.batch_size, args.z_size))
        z = torch.from_numpy(z).float()
        if args.gpus:
            z = z.cuda()

        # train generator
        if optimizer_idx == 0:
            # generate fake images
            self.generated_imgs = self(z) # apply forward pass

            # ground truth: generator wants the discrimator to think that its fake images are real
            valid = torch.ones(self.imgs.size(0), 1) # labels: vector of 1s, dimension: batch size
            valid = valid.type_as(self.imgs)

            # adversarial loss
            gen_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            self.log('gen_loss', gen_loss, on_epoch=True)
            return {'loss': gen_loss}
        
        # train discrimator
        if optimizer_idx == 1:
            # measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(self.imgs.size(0), 1)
            valid = valid.type_as(self.imgs)

            # discrimator output on real images
            real_output = self.discriminator(self.imgs.float())
            real_loss = self.adversarial_loss(real_output, valid)

            # how well can it label as fake?
            valid = torch.zeros(self.imgs.size(0), 1)
            valid = valid.type_as(self.imgs)

            # discriminator output on fake images
            fake_output = self.discriminator(self.generated_imgs.detach())
            fake_loss = self.adversarial_loss(fake_output, valid)

            # discrimator loss: average of fake and real loss
            disc_loss = (real_loss + fake_loss) / 2
            self.log('disc_loss', disc_loss, on_epoch=True)
            return {'loss': disc_loss}

    def configure_optimizers(self):
        lr = args.learning_rate
        b1 = args.beta_1
        b2 = args.beta_2

        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        return opt_gen, opt_disc


class SVHNCallbacks(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        self.fake_images = []
        self.real_images = []
       
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f'\nEpoch {epoch}:')
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')

        # save first 10 images of batch every 10 epochs
        num_images = 10
        if epoch % 10 == 0:

            print('gen images', pl_module.generated_imgs.shape)
            image_grid = make_grid(pl_module.generated_imgs.detach().cpu()[:num_images], nrow=5)
            img = image_grid.permute(1,2,0)
            img = ((img.numpy() +1)*255 / 2).astype(np.uint8)
            self.fake_images.append(img)

            image_grid = make_grid(pl_module.imgs[:num_images].cpu(), nrow=5)
            img = image_grid.permute(1,2,0)
            img = ((img.numpy() +1)*255 / 2).astype(np.uint8)
            self.real_images.append(img)

    def on_train_end(self, trainer, pl_module):
        print('final save', np.array(self.fake_images).shape)
        self.fake_images = np.stack(np.array(self.fake_images))
        self.real_images = np.stack(np.array(self.real_images))
        np.save(os.path.join(args.data_dir, 'fake_images_101e_dropout.npy'), self.fake_images)
        np.save(os.path.join(args.data_dir, 'real_images_101e_dropout.npy'), self.real_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data/SVHN/')
    # training params
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('-cd', '--conv-dim', type=int, default=64)
    parser.add_argument('-zs', '--z_size', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-b1', '--beta-1', type=float, default=0.5)
    parser.add_argument('-b2', '--beta-2', type=float, default=0.999)
    parser.add_argument('-nw', '--num-worker', type=int, default=1)
    parser.add_argument('-is', '--im-shape', type=int, nargs='+', default=[64, 64])
    parser.add_argument('-dd', '--disc-dropout', type=float, default=0.5)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print()

    data_module = SVHNDataModule(args)
    model = SVHNGAN(args)
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=201, progress_bar_refresh_rate=20,
                                            callbacks=[SVHNCallbacks()])
    trainer.fit(model, data_module)
    
