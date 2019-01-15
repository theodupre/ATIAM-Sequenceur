#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:12:19 2019

@author: theophile
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.utils import save_image

from src import vae_audio as audio
#%% Loading data
'''
Télechargement du dataset MNIST avec transformation des fichiers en Tensor et
binarisation des images
Les objets DataLoader permettent d'organiser les fichiers en batch et de pouvoir
y accéder facilement pour l'entraînement
'''
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

# Creation d'un dossier results/ pour stocker les resultats
results_dir = 'results/'
saving_dir = 'models/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

#%% Définition des modules du VAE : Encoder, Z_sampling, Decoder

"""
N : taille d'un batch
D_in : dimension d'entrée d'une donnée  
H_enc, H_dec : dimensions de la couche cachée du decoder et de l'encoder respectivement
D_out : dimension d'une donnée en sortie (= D_in)
D_z : dimension de l'epace latent
"""
#### Cyran's parameters of the death
# I'll make the code compatible with this fuckin shit, I'll send you a similar code in a while
conv1 = [1, 8, (11,5), (3,2), (5,2)]
conv2 = [8, 16, (11,5), (3,2), (5,2)]
conv3 = [16, 32, (11,5), (3,2), (5,2)]
conv = [conv1, conv2, conv3]

#The MLP hidden Layers : [[in_dim,hlayer1_dim], [hlayer1_dim,hlayer2_dim], ...]
enc_h_dims = [[229376, 1024], [1024, 512]]
dec_h_dims = [[32, 1024], [1024, 229376]]
#The Deconv Layers: [in_channels, out_channels, kernel_size, stride, padding, output_padding]
deconv1 = [32, 16, (11,5), (3,2), (5,2), (2,0)]
deconv2 = [16, 8, (11,8), (3,2), (5,2), (2,0)]
deconv3 = [8, 1, (11,4), (3,2), (5,2), (1,0)]
deconv = [deconv1, deconv2, deconv3]


N, input_size, D_z = batch_size, (28,28), 32
    
def train_vae(epoch,beta):
    train_loss = 0
    vae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        #data = data.reshape((-1,784))
        
        out_mu, out_var, latent_mu, latent_logvar = vae(data)
        loss = audio.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Affichage de la loss tous les 100 batchs
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
            #data = data.reshape((-1,784))
            
            out_mu, out_var, latent_mu, latent_logvar = vae(data)
            loss = audio.vae_loss(data, out_mu, out_var, latent_mu, latent_logvar, beta)
            test_loss += loss.item()
            
            # Sauvegarde d'exemples de données reconstituées avec les données d'origine
#            if batch_idx == 0:
#                n = min(data.size(0), 8)
#
#                comparison = torch.cat([data.view(N, 1, 28, 28)[:n],
#                                      out_mu.view(N, 1, 28, 28)[:n]])
#                save_image(comparison.cpu(),
#                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                
    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
    mean_test_loss.append(test_loss)
    
    return test_loss
            
    
#%% Instanciation du VAE

vae = audio.VAE_AUDIO(input_size, conv, enc_h_dims, D_z, deconv, dec_h_dims)
optimizer = torch.optim.Adam(vae.parameters(), 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#%% Training loop
beta = 4 # warm up coefficient 
num_epoch = 100
mean_train_loss = []
mean_test_loss = []
# Itération du modèle sur 50 epoches
for epoch in range(num_epoch):
    #beta += 1/num_epoch
    train_loss = train_vae(epoch,beta)
    _ = test_vae(epoch,beta)
    scheduler.step(train_loss)
    
    # Sauvegarde d'exemples de données générées à partir de l'espace latent
#    with torch.no_grad():
#        #sample = torch.randn(64, D_z)
#        Nd = 8
#        sample_x, sample_y = np.meshgrid(4*np.linspace(0,1,Nd)-2,4*np.linspace(0,1,Nd)-2)
#        sample_x = sample_x.reshape(Nd**2,1)
#        sample_y = sample_y.reshape(Nd**2,1)
#        sample = np.concatenate((sample_x,sample_y),axis=1)
#        sample = torch.from_numpy(sample).type(torch.float)
#        sample = vae.decoder(sample)
#        save_image(sample.view(Nd**2, 1, 28, 28), 
#'results/sample_' + str(epoch) + '.png')

#%% Saving model
import pickle

torch.save(vae.state_dict(), saving_dir + 'VAE_GAUSSIAN_10_BETA_4_hid800_2')   
loss = {"train_loss":mean_train_loss, "test_loss":mean_test_loss}  

with open(saving_dir + 'VAE_GAUSSIAN_10_BETA_4_hid800_loss2.pickle', 'wb') as handle:
    pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)     
