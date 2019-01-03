#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:56:32 2018

@author: theophile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%%
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

class VAE_GAUSSIAN(nn.Module):
    def __init__(self, D_in, D_enc, D_z, D_dec, D_out):
        super(VAE_GAUSSIAN,self).__init__()
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
#%%
D_in, D_enc, D_z, D_dec, D_out = 784, 512, 8, 512, 784
vae = VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out);
vae.load_state_dict(torch.load('models/VAE_BERNOULLI_8'))
vae.eval()



sample = torch.rand(1,8)
print(sample)
sample = vae.decoder(sample)
y = sample.detach().numpy()
for i in range(y.size):
    if y[0][i] > 0.5:
        y[0][i] = 1
    else:
        y[0][i] = 0
x = y.reshape(28,28) 
plt.figure()
plt.imshow(x, cmap='gray')