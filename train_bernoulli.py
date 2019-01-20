#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:06:47 2018

@author: theophile

Script to train the bernoulli vae

First, the dataset containing activation matrices is loaded.
Then, the parameters of the network are instanciated.
The train and test method are defined, the vae is instanciated
Finally, the training process is launched and the weights and the loss 
are stored.

The latent space is contrained to be gaussian (i.e. mean and variance)
The output is constrained to be bernoulli distributed because the data is
composed of discrete values {0,1}

"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler

from src import vae_bernoulli as bernoulli
from src import DatasetLoader as dataset

#%% Loading data

batch_size = 8
data_dir = 'Dataset_Drum_Groove_Pattern/'
dataset = dataset.DatasetLoader(data_dir,transform=True)

test_split = .2
shuffle_dataset = True
random_seed= 42
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


# Creation of a folder to store model weights
saving_dir = 'models/sequence/'
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

#%% Parmeters of the network

"""
N : batch_size
D_in : input dimension
H_enc, H_dec : Hidden layer size (encoder and decoder respectively)
D_out : output dimension (= D_in)
D_z : latent space dimension
"""
N, D_in, D_enc, D_z, D_dec, D_out = batch_size, 512, 800, 5, 800, 512


#%% Train and test method

mean_train_loss = []
def train_vae(epoch,beta):
    train_loss = 0
    vae.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.reshape((-1,512))
        x_approx, mu, var = vae(data)
        loss = bernoulli.vae_loss(data, x_approx, mu, var, beta)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Display of loss every 10 batches
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss*N / len(train_loader.dataset)))
    mean_train_loss.append(train_loss*N / len(train_loader.dataset))
        
mean_test_loss = []        
def test_vae(epoch,beta):
    test_loss = 0
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.reshape((-1,512))
            
            x_approx, mu, var = vae(data)
            loss = bernoulli.vae_loss(x_approx, data, mu, var, beta)
            test_loss += loss.item()
            
            # Sauvegarde d'exemples de données reconstituées avec les données d'origine
            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(N, 1, 8, 64)[:n],
                                      x_approx.view(N, 1, 8, 64)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
    mean_test_loss.append(test_loss)
            
    
#%% Instanciation du VAE

vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out)
optimizer = torch.optim.Adam(vae.parameters(), 1e-3)

#%% Training loop
beta = 4 # warm up coefficient 
num_epoch = 200
# Itération du modèle sur 50 epoches
for epoch in range(num_epoch):
    
    train_vae(epoch,beta)
    test_vae(epoch,beta)

#%% Saving model
import pickle

torch.save(vae.state_dict(), saving_dir + 'VAE_BERNOULLI_5_BETA_4_hid800')   
loss = {"train_loss":mean_train_loss, "test_loss":mean_test_loss}  

with open(saving_dir + 'VAE_BERNOULLI_5_BETA_4_hid800_loss.pickle', 'wb') as handle:
    pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


