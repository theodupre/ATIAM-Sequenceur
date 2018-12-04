#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:34:36 2018

@author: theophile

30/11 : Code du VAE en pytorch, s'execute sans erreur mais ne semble pas fonctionner 
        correctement (loss converge vers -10 ?)
        En plus l'image reconstituée est toujours identique (une sorte de 9 très floue)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from torchvision.utils import save_image

#%% Loading data
'''
Télechargement du dataset MNIST avec transformation des fichiers en Tensor et
normalisation (je ne sais pas vraiment ce que fait la normalisation)
Les objets DataLoader permettent d'organiser les fichiers en batch et de pouvoir
y accéder facilement pour l'entraînement
Pour l'instant, le test_dataset n'est pas utilisé
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
if not os.path.exists(results_dir):
 os.makedirs(results_dir)

#%% Définition des modules du VAE : Encoder, Z_sampling, Decoder

"""
N : taille d'un batch
D_in : dimension d'entrée d'une donnée  
H_enc, H_dec : dimensions de la couche cachée du decoder et de l'encoder respectivement
D_out : dimension d'une donnée en sortie (= D_in)
D_z : dimension de l'epace latent
"""
N, D_in, D_enc, D_z, D_dec, D_out = batch_size, 784, 512, 2, 512, 784


class VAE(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE,self).__init__()
        self.linearEnc = nn.Linear(D_in, D_enc)
        self.linearMu = nn.Linear(D_enc, D_z)
        self.linearVar = nn.Linear(D_enc, D_z)
        self.linearDec = nn.Linear(D_z, D_dec)
        self.linearOut = nn.Linear(D_dec, D_out)
        
    def forward(self,x):
        mu, logvar = self.encoder(x)
        z_sample = self.reparametrize(mu, logvar)
        x_approx = self.decoder(z_sample)
        
        return x_approx, mu, logvar
    
    def encoder(self,x):
        
        h_relu = F.relu(self.linearEnc(x))
        mu = self.linearMu(h_relu)
        logvar = self.linearVar(h_relu)
        
        return mu, logvar
    
    def reparametrize(self, mu,logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad = True)
        
        return eps.mul(std).add_(mu)
    
    def decoder(self, z_sample):
        
        h_relu = F.relu(self.linearDec(z_sample))
        h_out = F.sigmoid(self.linearOut(h_relu))
        
        return h_out 

def vae_loss(x, x_sample, mu, logvar):
    recons_loss = nn.BCELoss(reduction='sum')
    KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recons_loss(x_sample, x) + KL_div
    
def train_vae(epoch):
    train_loss = 0
    vae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.reshape((-1,784))
        
        x_approx, mu, logvar = vae(data)
        loss = vae_loss(data, x_approx, mu, logvar)
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
        
        
def test_vae(epoch):
    test_loss = 0
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.reshape((-1,784))
            
            x_approx, mu, logvar = vae(data)
            loss = vae_loss(x_approx, data, mu, logvar)
            test_loss += loss.item()
            
            # Sauvegarde d'exemples de données reconstituées avec les données d'origine
            if batch_idx == 0:
                n = min(data.size(0), 8)

                comparison = torch.cat([data.view(N, 1, 28, 28)[:n],
                                      x_approx.view(N, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)/N
    print('====> Test set loss: {:.4f}'.format(test_loss))
            
    
#%% Instanciation du VAE

vae = VAE(D_in, D_enc, D_z, D_dec, D_out)
optimizer = torch.optim.Adam(vae.parameters(), 1e-3)

#%% Training loop

# Itération du modèle sur 50 epoches
for epoch in range(50): 
    train_vae(epoch)
    test_vae(epoch)
    
    # Sauvegarde d'exemples de données générées à partir de l'espace latent
    with torch.no_grad():
        #sample = torch.randn(64, D_z)
        Nd = 8
        sample_x, sample_y = np.meshgrid(4*np.linspace(0,1,Nd)-2,4*np.linspace(0,1,Nd)-2)
        sample_x = sample_x.reshape(Nd**2,1)
        sample_y = sample_y.reshape(Nd**2,1)
        sample = np.concatenate((sample_x,sample_y),axis=1)
        sample = torch.from_numpy(sample).type(torch.float)
        sample = vae.decoder(sample)
        save_image(sample.view(Nd**2, 1, 28, 28), 
'results/sample_' + str(epoch) + '.png')
      