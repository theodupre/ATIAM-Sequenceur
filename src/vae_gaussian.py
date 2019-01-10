#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:39:12 2019

@author: theophile
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_GAUSSIAN(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE_GAUSSIAN,self).__init__()
        self.linearEnc = nn.Linear(D_in, D_enc)
        self.linearMu = nn.Linear(D_enc, D_z)
        self.linearVar = nn.Linear(D_enc, D_z)
        self.linearDec = nn.Linear(D_z, D_dec)
        self.linearOutMu = nn.Linear(D_dec, D_out)
        self.linearOutVar = nn.Linear(D_dec, D_out)
        
    def forward(self,x):
        latent_mu, latent_logvar = self.encoder(x)
        z_sample = self.reparametrize(latent_mu, latent_logvar)
        out_mu, out_var = self.decoder(z_sample)
        
        return out_mu, out_var, latent_mu, latent_logvar
    
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
        h_out_mu = self.linearOutMu(h_relu)
        h_out_var = self.linearOutVar(h_relu)
        
        return h_out_mu, h_out_var 
   
def vae_loss(x, out_mu, out_var, latent_mu, latent_var, beta):
    
    recons_loss = -torch.sum(0.5*torch.log(2*3.14*out_var) + (x - out_mu)**2/out_var)
    KL_div = -0.5 * torch.sum(1 + latent_var - latent_mu.pow(2) - latent_var.exp())

    return recons_loss + beta*KL_div