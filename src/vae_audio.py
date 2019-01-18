#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:45:43 2019

@author: theophile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.convLayer import convNN,deconvNN


class VAE_AUDIO(nn.Module):
    def __init__(self, input_size, convParams, encMlpParams, D_z, deconvParams, decMlpParams):
        super(VAE_AUDIO,self).__init__()
        self.conv = convNN(convParams)
        self.deconv = deconvNN(deconvParams)
        
        self.encMlp1 = nn.Linear(encMlpParams[0][0],encMlpParams[0][1])
        self.encMlp2 = nn.Linear(encMlpParams[1][0],encMlpParams[1][1])
        self.decMlp1 = nn.Linear(decMlpParams[0][0]+8,decMlpParams[0][1])
        self.decMlp2 = nn.Linear(decMlpParams[1][0],decMlpParams[1][1])
        
        self.latentMu = nn.Linear(encMlpParams[1][1], D_z)
        self.latentVar = nn.Linear(encMlpParams[1][1], D_z)
        self.outMu = nn.Linear(1,1)#input_size[0]*input_size[1], input_size[0]*input_size[1])
        self.outVar = nn.Linear(1,1)#input_size[0]*input_size[1], input_size[0]*input_size[1])

    def forward(self,x, label):
        latent_mu, latent_logvar = self.encoder(x)
        z_sample = self.reparametrize(latent_mu, latent_logvar)
        out_mu, out_var = self.decoder(z_sample,label)
        
        return latent_mu, latent_logvar, out_mu, out_var
    
    def encoder(self,x):
        
        h_conv, = self.conv(x)
        h_conv_flat = h_conv.view(-1,10240)
        h_mlp = F.relu(self.encMlp2(F.relu(self.encMlp1(h_conv_flat))))
        latent_mu = self.latentMu(h_mlp)
        latent_var = self.latentVar(h_mlp)
        
        return latent_mu, latent_var
    
    def reparametrize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad = True)
        
        return eps.mul(std).add_(mu)
    
    def decoder(self, z_sample, label):
        print('z', z_sample.shape, 'label', label.shape)
        dec_input = torch.cat((z_sample, label), 1)
        h_mlp = F.relu(self.decMlp2(F.relu(self.decMlp1(dec_input))))
        h_mlp = h_mlp.view(-1,32,16,20)
        print(h_mlp.shape)
        h_deconv = self.deconv(h_mlp)
        h_deconv = h_deconv.view(-1,410*157)
        print(h_deconv.shape)
        out_mu = self.outMu(h_deconv)
        out_var = self.outVar(h_deconv)
        
        return out_mu, out_var
    
def vae_loss(x, out_mu, out_logvar, latent_mu, latent_logvar, beta):
    
    recons_loss = torch.sum(0.5*1.837369980 + 0.5*out_logvar + (x - out_mu)**2/(2*out_logvar.exp()))
    KL_div = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

    return recons_loss + beta*KL_div
