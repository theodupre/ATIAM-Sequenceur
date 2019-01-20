#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:12:19 2019

@author: theophile

Script to train the convolutional vae

First, the dataset containing drum sound transform with the gabor transform
is loaded.
Then, the parameters of the network are instanciated.
The train and test method are defined, the vae is instanciated
Finally, the training process is launched and the weights and the loss 
are stored.

The latent space is contrained to be gaussain (i.e. mean and variance)
The output is also constrained to be gaussian because the data is composed of 
real values 

REQUIREMENT : the dataset is not on the git repository, please contact us 
if you want it 

dupre@atiam.fr

"""


import os
import pickle
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from src import vae_audio as audio
from src import DatasetLoader as dataset

#%% Loading data
batch_size = 200
data_dir = 'data/dataset_audio/'
dataset = dataset.DatasetLoader(data_dir,transform=True, audio=True)

test_split = .2
shuffle_dataset = True

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

# Create directory to store model weights
saving_dir = 'models/audio/'
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

#%% Definition of the network parameters

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#### Cyran's parameters of the death
#The Conv Layers: [in_channels, out_channels, kernel_size, stride, padding]
conv1 = [1, 8, (11,5), (3,2), (5,2), (2,2)]
conv2 = [8, 16, (11,5), (3,2), (5,2), (2,2)]
conv3 = [16, 32, (11,5), (3,2), (5,2), (2,2)]
conv = [conv1, conv2, conv3]

#The MLP hidden Layers : [[in_dim,hlayer1_dim], [hlayer1_dim,hlayer2_dim]]
enc_h_dims = [[10240, 1024], [1024, 512]]
dec_h_dims = [[128, 1024], [1024, 10240]]

#The Deconv Layers: [in_channels, out_channels, kernel_size, stride, padding, output_padding]
deconv1 = [32, 16, (11,5), (3,2), (5,2), (2,2), (0,1,0,0)]
deconv2 = [16, 8, (11,5), (3,2), (5,2), (2,2), (0,0,1,0)]
deconv3 = [8, 1, (11,5), (3,2), (5,2), (2,2), (0,0,1,0)]
deconv = [deconv1, deconv2, deconv3]

N, input_size, D_z = batch_size, (410,157), 128 # batch_size, input-size, latent space dimension
#%%
# Train method
def train_vae(epoch,beta):
    train_loss = 0
    vae.train()
    for batch_idx, (data, label) in enumerate(train_loader):

        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        out_mu, out_var, latent_mu, latent_logvar = vae(data, label) # forward
        loss = audio.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Affichage de la loss tous les 100 batchs
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss*N / len(train_loader.dataset)))
    mean_train_loss.append(train_loss*N / len(train_loader.dataset))
    return train_loss

#Test method
def test_vae(epoch,beta):
    test_loss = 0
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)

            out_mu, out_var, latent_mu, latent_logvar = vae(data, label) # forward
            loss = audio.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
    mean_test_loss.append(test_loss)

    return test_loss


#%% Instanciation du VAE

vae = audio.VAE_AUDIO(input_size, conv, enc_h_dims, D_z, deconv, dec_h_dims).to(device)
optimizer = torch.optim.Adam(vae.parameters(), 1e-4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#%% Training loop
beta = 4 # warm up coefficient
num_epoch = 500
mean_train_loss = []
mean_test_loss = []
# Itération du modèle sur 50 epoches
for epoch in range(num_epoch):

    train_loss = train_vae(epoch,beta)
    _ = test_vae(epoch,beta)
    scheduler.step(train_loss)

    if epoch % 10 == 0 and epoch != 0:
        torch.save(vae.state_dict(), saving_dir + 'VAE_AUDIO_128_epoch' + str(epoch))
        loss = {"train_loss":mean_train_loss, "test_loss":mean_test_loss}

        with open(saving_dir + 'VAE_AUDIO_128_loss_epoch' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
