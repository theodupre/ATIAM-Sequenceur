#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:20:07 2019

@author: theophile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAE_BERNOULLI(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE_BERNOULLI,self).__init__()
        self.linearEnc = nn.Linear(D_in, D_enc)
        self.linearMu = nn.Linear(D_enc, D_z)
        self.linearDec = nn.Linear(D_z, D_dec)
        self.linearOut = nn.Linear(D_dec, D_out)
        self.batchNorm = nn.BatchNorm1d(D_enc)
        
    def forward(self,x):
        mu = self.encoder(x)
        z_sample = self.reparametrize(mu)
        x_approx = self.decoder(z_sample)
        
        return x_approx, mu
    
    def encoder(self,x):
        
        h_norm = self.batchNorm(self.linearEnc(x))
        h_relu = F.relu(h_norm)
        mu = F.sigmoid(self.linearMu(h_relu))
        
        return mu
    
    def reparametrize(self, mu):
        
        eps = torch.rand_like(mu, requires_grad = True)
        
        return F.sigmoid(torch.log(eps + 1e-20) - torch.log(1 - eps + 1e-20) + torch.log(mu + 1e-20) 
                         - torch.log(1 - mu + 1e-20))
    
    def decoder(self, z_sample):
        
        h_relu = F.relu(self.linearDec(z_sample))
        h_out = F.sigmoid(self.linearOut(h_relu))
        
        return h_out 
    
def vae_loss(x, x_sample, mu, beta):
    
    p = 0.5
    recons_loss = nn.BCELoss(reduction='sum')
    KL_div = torch.mul(mu, torch.log(mu + 1e-20) - np.log(p)) + torch.mul(1 - mu, torch.log(1 - mu + 1e-20) - np.log(1 - p))

    return recons_loss(x_sample, x) + beta*torch.sum(KL_div)
