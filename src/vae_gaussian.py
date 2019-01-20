#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:39:12 2019

@author: theophile

Class defining the gaussian vae
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_GAUSSIAN(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE_GAUSSIAN,self).__init__()
        # Encoder hidden layer
        self.linearEnc = nn.Linear(D_in, D_enc)
        # Latent space layer
        self.linearLatentMu = nn.Linear(D_enc, D_z)
        self.linearLatentVar = nn.Linear(D_enc, D_z)
        # Decoder hidden layer
        self.linearDec = nn.Linear(D_z, D_dec)
        # Output layer
        self.linearOutMu = nn.Linear(D_dec, D_out)
        self.linearOutVar = nn.Linear(D_dec, D_out)
        
    def forward(self,x):
        # Encoder
        latent_mu, latent_logvar = self.encoder(x)
        # Reparametriaztion trick
        z_sample = self.reparametrize(latent_mu, latent_logvar)
        # Decoder
        out_mu, out_logvar = self.decoder(z_sample)
        
        return out_mu, out_logvar, latent_mu, latent_logvar
    
    def encoder(self,x):
        
        h_relu = F.relu(self.linearEnc(x))
        mu = self.linearLatentMu(h_relu)
        logvar = self.linearLatentVar(h_relu)
        
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        # Sampling in the latent space to allow backpropagation
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad = True)
        
        return eps.mul(std).add_(mu)
    
    def decoder(self, z_sample):
        
        h_relu = F.relu(self.linearDec(z_sample))
        mu = self.linearOutMu(h_relu)
        logvar = self.linearOutVar(h_relu)
        
        return mu, logvar 

def vae_loss(x, out_mu, out_logvar, latent_mu, latent_logvar, beta):
    # log(P(x|z,mu(x),var(x)))
    recons_loss = torch.sum(0.5*1.837369980 + 0.5*out_logvar + (x - out_mu)**2/(2*out_logvar.exp()))
    # KL divergence between gaussian prior and latent space distribution
    KL_div = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

    return recons_loss + beta*KL_div