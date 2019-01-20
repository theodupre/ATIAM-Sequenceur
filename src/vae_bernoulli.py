#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:20:07 2019

@author: theophile

Class defining the bernoulli vae

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_BERNOULLI(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE_BERNOULLI,self).__init__()
        # Encoder hidden layer
        self.linearEnc = nn.Linear(D_in, D_enc)
        
        # latent space layer
        self.linearMu = nn.Linear(D_enc, D_z)
        self.linearVar = nn.Linear(D_enc, D_z)
        
        # Decoder hidden layer
        self.linearDec = nn.Linear(D_z, D_dec)
        
        # Output layer 
        self.linearOut = nn.Linear(D_dec, D_out)
        
    def forward(self,x):
        # Encoder
        mu, logvar = self.encoder(x)
        # Reparametrization trick
        z_sample = self.reparametrize(mu, logvar)
        # Decoder
        x_approx = self.decoder(z_sample)
        
        return x_approx, mu, logvar
    
    def encoder(self,x):
        
        h_relu = F.relu(self.linearEnc(x))
        mu = self.linearMu(h_relu)
        var = self.linearVar(h_relu)
        
        return mu, var
    
    def reparametrize(self, mu, logvar):
        # Sampling in the latent space to allow backpropagation
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad = True)
        
        return eps.mul(std).add_(mu)
    
    def decoder(self, z_sample):
        
        h_relu = F.relu(self.linearDec(z_sample))
        h_out = F.sigmoid(self.linearOut(h_relu))
        
        return h_out 
    
def vae_loss(x, x_sample, mu, logvar, beta):
    # Binary cross entropy
    recons_loss = nn.BCELoss(reduction='sum')
    # KL divergence between gaussian prior and latent space distribution
    KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recons_loss(x_sample, x) + beta*torch.sum(KL_div)
