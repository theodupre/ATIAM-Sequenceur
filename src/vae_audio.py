#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:45:43 2019

@author: theophile

Class defining the convolutional vae

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.convLayer import convNN,deconvNN


class VAE_AUDIO(nn.Module):
    def __init__(self, input_size, convParams, encMlpParams, D_z, deconvParams, decMlpParams):
        super(VAE_AUDIO,self).__init__()
        
        # COnv and deconv layers
        self.conv = convNN(convParams)
        self.deconv = deconvNN(deconvParams)
        
        # Encoder Mlp layers 
        self.encMlp1 = nn.Linear(encMlpParams[0][0],encMlpParams[0][1])
        self.encMlp2 = nn.Linear(encMlpParams[1][0],encMlpParams[1][1])
        
        # Decoder Mlp Layers, first one has latent_dim + 8 input dim 
        # to condition the decoder depending on the decoded percussion
        self.decMlp1 = nn.Linear(decMlpParams[0][0]+8,decMlpParams[0][1])
        self.decMlp2 = nn.Linear(decMlpParams[1][0],decMlpParams[1][1])
        
        # Latent space layer
        self.latentMu = nn.Linear(encMlpParams[1][1], D_z)
        self.latentVar = nn.Linear(encMlpParams[1][1], D_z)
        
        # Output layer
        self.outConvMu = nn.Conv2d(1,1,(11,5), padding=(5,2))
        self.outConvVar = nn.Conv2d(1,1,(11,5), padding=(5,2))
        
        self.output_size = input_size[0]*input_size[1]
    
    def forward(self,x, label):
        # Encoding
        latent_mu, latent_logvar = self.encoder(x)
        # Reparametrization trick
        z_sample = self.reparametrize(latent_mu, latent_logvar)
        # Decoding
        out_mu, out_var = self.decoder(z_sample,label)
        
        return out_mu, out_var, latent_mu, latent_logvar
    
    def encoder(self,x):
        #Conv layer
        h_conv = self.conv(x)
        h_conv_flat = h_conv.view(-1,10240)
        #Mlp 
        h_mlp = F.relu(self.encMlp2(F.relu(self.encMlp1(h_conv_flat))))
        #Latent layer
        latent_mu = self.latentMu(h_mlp)
        latent_var = self.latentVar(h_mlp)
        
        return latent_mu, latent_var
    
    def reparametrize(self, mu, logvar):
        # Sampling from normal distribution
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad = True)
        
        return eps.mul(std).add_(mu)
    
    def decoder(self, z_sample, label):
        # Concatenation of latent_space sampling and label of data (kick, snare
        # etc) to condition the decoder
        dec_input = torch.cat((z_sample, label), 1)
        #Mlp
        h_mlp = F.relu(self.decMlp2(F.relu(self.decMlp1(dec_input))))
        h_mlp = h_mlp.view(-1,32,16,20)
        #Deconv layer
        h_deconv = self.deconv(h_mlp)
        #Output layer
        out_mu = self.outConvMu(h_deconv)
        out_var = self.outConvVar(h_deconv)
        
        return out_mu, out_var
    
def vae_loss(x, out_mu, out_logvar, latent_mu, latent_logvar, beta):
    # log(P(x|z,mu(x),var(x)))
    recons_loss = torch.sum(0.5*1.837369980 + 0.5*out_logvar + (x - out_mu)**2/(2*out_logvar.exp()))
    # KL divergence between gaussian prior and latent space distribution
    KL_div = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

    return recons_loss + beta*KL_div
