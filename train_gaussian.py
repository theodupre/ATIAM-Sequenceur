#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:34:36 2018

@author: theophile

Script to train the gaussian vae

First, the MNIST dataset is loaded.
Then, the parameters of the network are instanciated.
The train and test method are defined, the vae is instanciated
Finally, the training process is launched and the weights and the loss 
are stored.

The latent space is contrained to be gaussian (i.e. mean and variance)
The output is constrained to be gaussian because the data is
composed of real values (i.e. pixel value)

"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.utils import save_image

from src import vae_gaussian as gaussian



#%% Loading data

batch_size = 512
data_dir = 'data'
train_dataset = datasets.MNIST(data_dir, train=True, download=True, 
                    transform=transforms.Compose([transforms.ToTensor(),
                    lambda x: x > 0, # binarisation de l'image
                    lambda x: x.float()]))
test_dataset = datasets.MNIST(data_dir, train=False, download=True, 
                    transform=transforms.Compose([transforms.ToTensor(),
                    lambda x: x > 0, # binarisation de l'image
                    lambda x: x.float()]))
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)

# Creation of dir to store model weights and to store results
results_dir = 'results/MNIST/'
saving_dir = 'models/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
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

N, D_in, D_enc, D_z, D_dec, D_out = batch_size, 784, 800, 5, 800, 784

#%% Train and test method

def train_vae(epoch,beta):
    train_loss = 0
    vae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.reshape((-1,784))
        
        out_mu, out_var, latent_mu, latent_logvar = vae(data)
        loss = gaussian.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 batches
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss*N / len(train_loader.dataset)))
    mean_train_loss.append(train_loss*N / len(train_loader.dataset))
    return train_loss
        
                
def test_vae(epoch,beta):
    test_loss = 0
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.reshape((-1,784))
            
            out_mu, out_var, latent_mu, latent_logvar = vae(data)
            loss = gaussian.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
            test_loss += loss.item()
            
            # Sauvegarde d'exemples de données reconstituées avec les données d'origine
            if batch_idx == 0:
                n = min(data.size(0), 8)

                comparison = torch.cat([data.view(N, 1, 28, 28)[:n],
                                      out_mu.view(N, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                
    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
    mean_test_loss.append(test_loss)
    
    return test_loss
            
    
#%% Instanciation du VAE

vae = gaussian.VAE_GAUSSIAN(D_in, D_enc, D_z, D_dec, D_out)
optimizer = torch.optim.Adam(vae.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#%% Training loop
beta = 1 # warm up coefficient 
num_epoch = 100
mean_train_loss = []
mean_test_loss = []
# Itération du modèle sur 50 epoches
for epoch in range(num_epoch):

    train_loss = train_vae(epoch,beta)
    _ = test_vae(epoch,beta)
    scheduler.step(train_loss)
    
#%% Saving model
import pickle

torch.save(vae.state_dict(), saving_dir + 'VAE_GAUSSIAN_10_BETA_1_hid800')   
loss = {"train_loss":mean_train_loss, "test_loss":mean_test_loss}  

with open(saving_dir + 'VAE_GAUSSIAN_10_BETA_1_hid800.pickle', 'wb') as handle:
    pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)     
