#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:34:36 2018

@author: theophile

30/11 : Code du VAE en pytorch, s'execute sans erreur mais ne semble pas fonctionner 
        correctement (loss converge vers -10 ?)
        En plus l'image reconstituée est toujours identique (une sorte de 9 très floue)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

#%% Loading data
'''
Télechargement du dataset MNIST avec transformation des fichiers en Tensor et
normalisation (je ne sais pas vraiment ce que fait la normalisation)
Les objets DataLoader permettent d'organiser les fichiers en batch et de pouvoir
y accéder facilement pour l'entraînement
Pour l'instant, le test_dataset n'est pas utilisé
'''
batch_size = 10
data_dir = 'data'
train_dataset = datasets.MNIST(data_dir, train=True, download=True, 
                    transform=transforms.Compose([transforms.ToTensor(),
                    lambda x: x > 0, # binarisation de l'image
                    lambda x: x.float(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    ]))
test_dataset = datasets.MNIST(data_dir, train=False, download=True, 
                    transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)

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
        eps = torch.randn_like(std)
        
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
    
    vae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.reshape((-1,784))
        
        x_approx, mu, logvar = vae(data)
        loss = vae_loss(data, x_approx, mu, logvar)
        loss.backward()
        optimizer.step()
        
        # Affichage de la loss tous les 100 batchs
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            display_data(batch_size - 1, x_approx, data)
        
        
def test_vae(epoch):
    
    vae.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data.reshape((-1,784))
            
            x_approx, mu, logvar = vae(data)
            loss = vae_loss(x_approx, data, mu, logvar)
            
            # Affichage de la loss tous les 100 batchs
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                display_data(batch_size - 1, x_approx, data)
                
    
'''
Fonction d'affichage d'une image et de sa reconstruction
@param: data_id: id entre 0 et batch_size
'''
def display_data(data_id, x_approx, data):
        approx_data = x_approx[data_id].detach().numpy().reshape((28,28))
        data = data[data_id].detach().numpy().reshape((28,28))
        plt.figure
        plt.subplot(1,2,1)
        plt.imshow(data, cmap='gray')
        plt.xlabel('Original data')
        plt.subplot(1,2,2)
        plt.imshow(approx_data, cmap='gray')
        plt.xlabel('Reconstructed data')        
    


#%% Instanciation du VAE

vae = VAE(D_in, D_enc, D_z, D_dec, D_out)
optimizer = torch.optim.Adam(vae.parameters(), 1e-4)

#%% Training loop

# Itération du modèle sur 5 epoches
for epoch in range(5): 
    
    
    train_vae(epoch)
    test_vae(epoch)
    
    
    '''
    # parcourt de tous les batchs à chaque epoch
    for batch_idx, (data, target) in enumerate(train_loader): 
        # target n'est pas utilisé car la loss s'évalue par rapport à la donnée 
        # de départ et non par rapport à la classe de la donnée
        X = data.reshape((N,D_in)) # Image (28,28) tranformée en vecteur (784,1)
        mu, var = encoder.forward(X) # Encodage
        z = sample_z(mu, var) # Tirage d'une valeur z dans l'espace latent
        X_sample = decoder.forward(z) # Décodage
        
        # Loss = erreur de reconstruction de la donnée + KL-divergence
        # KL-div : Calcul de la ressemblance entre la distribution réel P(X|Z) (intraçable)
        #           et de son approximation Q(X|Z) calculée par le vae
        kl_loss = 0.5 * torch.sum(torch.exp(var) + mu**2 - 1. - var) 
        loss = loss_recons(X_sample, X) + kl_loss
        
        # Mise à zero des gradients 
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        
        # Retropropagation
        loss.backward()
        
        # Mise à jour des poids
        optimizer_enc.step()
        optimizer_dec.step()
        
        # Affichage de la loss tous les 100 batchs
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            display_data(batch_size - 1)
    '''    