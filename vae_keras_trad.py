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
N, D_in, H_enc, D_z, H_dec, D_out = batch_size, 784, 512, 2, 512, 784

"""Module de l'encoder Q(Z|X) : une couche cachée Linear + Relu et deux couches de sorties Linear
@param :    D_in : Dimension en entrée 
            H_enc = Dimension de la couchée 
            D_z : Dimension de l'espace latent et de sortie du decoder

"""
class Encoder(nn.Module):
    def __init__(self, D_in, H_enc, D_z):
        """
        Instanciation des modules nécessaires pour le décodeur dans le constructeur
        La fonction d'activation Relu n'est pas instancier ici car elle ne contient pas 
        de paramètres pour la retropropagation
        """
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H_enc)
        self.linear_mu = nn.Linear(H_enc, D_z)
        self.linear_var = nn.Linear(H_enc, D_z)

    def forward(self, x):
        """
        Assemblage des modules : 
            Linear -> Relu  -> Linear -> mu
                            -> Linear -> var
                            
        Le decodeur renvoit la moyenne mu et la variance var
        """
        h_relu = F.relu(self.linear1(x))
        mu = self.linear_mu(h_relu)
        var = self.linear_var(h_relu)
        
        return mu, var

"""Module du Decoder P(X|Z) : une couche cachée Linear + Relu et une couche de sortie Linear + sigmoid
@param :    D_z : Dimension en entrée du decoder qui correspond aux dimensions de l'espace latent
            H_dec = Dimensions de la couchée cachée
            D_out : Dimensions de l'espace latent de sortie (taille de l'image reconstruite)

"""
class Decoder(nn.Module):
    def __init__(self, D_z, H_dec, D_out):
        """
        Instanciation des modules du decoder : 
        """
        super(Decoder, self).__init__()
        self.linear_hidden = nn.Linear(D_z, H_dec)
        self.linear_out = nn.Linear(H_dec, D_out)

    def forward(self, x):
        """
        Assemblage des modules :
            Linear -> Relu -> Linear -> Sigmoid -> image en sortie
        
        Retourne l'image reconstruite
        """
        h_relu = F.relu(self.linear_hidden(x))
        h_out = F.sigmoid(self.linear_out(h_relu))
        
        return h_out

'''
Fonction d'affichage d'une image et de sa reconstruction
@param: data_id: id entre 0 et batch_size
'''
def display_data(data_id):
        approx_data = X_sample[data_id].detach().numpy().reshape((28,28))
        data = X[data_id].detach().numpy().reshape((28,28))
        plt.figure
        plt.subplot(1,2,1)
        plt.imshow(data, cmap='gray')
        plt.xlabel('Original data')
        plt.subplot(1,2,2)
        plt.imshow(approx_data, cmap='gray')
        plt.xlabel('Reconstructed data')        
    
"""
Reparametrization trick.  
Tirage d'un valeur z dans la distribution N(mu,var) afin de pouvoir décoder des valeurs 
déterministes (i.e. une image) et non une distribution de probabilité d'image. 
"""        
def sample_z(mu, log_sigma):
    eps = torch.randn(N,D_z)
    z_sample = mu + torch.exp(log_sigma / 2) * eps
    return z_sample


#%% Instanciation du VAE

# Instanciation des modules
encoder = Encoder(D_in, H_enc, D_z)
decoder = Decoder(D_z, H_dec, D_out)

# Instanciation des optimizer (i.e. mise à jour des poids)
optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=1e-4)
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=1e-4)

# Instanciation de la Loss (seulement la partie évaluation de la recontruction)
loss_recons = nn.BCELoss()

#%% Training loop

# Itération du modèle sur 20 epoch
for epoch in range(10): 
    
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
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            display_data(batch_size - 1)





