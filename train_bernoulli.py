#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:06:47 2018

@author: theophile
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.utils import save_image

from src import vae_bernoulli as bernoulli
from src import DatasetLoader as data

#%% Loading data
'''
Télechargement du dataset MNIST avec transformation des fichiers en Tensor et
normalisation (je ne sais pas vraiment ce que fait la normalisation)
Les objets DataLoader permettent d'organiser les fichiers en batch et de pouvoir
y accéder facilement pour l'entraînement
Pour l'instant, le test_dataset n'est pas utilisé
'''
batch_size = 8
data_dir = 'data/dataset_sequence/'
train_dataset = data.DatasetLoader(data_dir,transform=True)
test_dataset = train_dataset

#train_dataset = datasets.MNIST(data_dir, train=True, download=True, 
#                    transform=transforms.Compose([transforms.ToTensor(),
#                    lambda x: x > 0, # binarisation de l'image
#                    lambda x: x.float()]))
#test_dataset = datasets.MNIST(data_dir, train=False, download=True, 
#                    transform=transforms.Compose([transforms.ToTensor(),
#                    lambda x: x > 0, # binarisation de l'image
#                    lambda x: x.float()]))
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)

# Creation d'un dossier results/ pour stocker les resultats et d'un dossier models/ pour sauvegarder les paramètres
results_dir = 'results/'
saving_dir = 'models/sequence/'
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
N, D_in, D_enc, D_z, D_dec, D_out = batch_size, 1024, 512, 10, 512, 1024


def train_vae(epoch,beta):
    train_loss = 0
    vae.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.reshape((-1,1024))
        x_approx, mu = vae(data)
        loss = bernoulli.vae_loss(data, x_approx, mu, beta)
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
        
        
def test_vae(epoch,beta):
    test_loss = 0
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.reshape((-1,1024))
            
            x_approx, mu = vae(data)
            loss = bernoulli.vae_loss(x_approx, data, mu, beta)
            test_loss += loss.item()
            
            # Sauvegarde d'exemples de données reconstituées avec les données d'origine
            if batch_idx == 0:
                n = min(data.size(0), 8)

                comparison = torch.cat([data.view(N, 1, 8, 128)[:n],
                                      x_approx.view(N, 1, 8, 128)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
            
    
#%% Instanciation du VAE

vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out)
optimizer = torch.optim.Adam(vae.parameters(), 1e-3)

#%% Training loop
beta = 4 # warm up coefficient 
num_epoch = 200
# Itération du modèle sur 50 epoches
for epoch in range(num_epoch):
    #beta += 1/num_epoch
    train_vae(epoch,beta)
    test_vae(epoch,beta)

#%% Saving model
        
torch.save(vae.state_dict(), saving_dir + 'VAE_BERNOULLI_10_BETA_4')        

