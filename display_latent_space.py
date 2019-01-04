#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:56:32 2018

@author: theophile
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

from src import vae_gaussian as gaussian
from src import vae_bernoulli as bernoulli
   
#%% Load vae, you can choose between gaussain and bernoulli models
D_in, D_enc, D_z, D_dec, D_out = 784, 512, 2, 512, 784 # Change D_z depending on dimensions of latent space
vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out); # change 'gaussian' to 'bernoulli' to change the model
vae.load_state_dict(torch.load('models/VAE_BERNOULLI_2_BETA_4')) # idem 
vae.eval()

#%% Sampling from latent space
sample = torch.rand(1,D_z)
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

#%% Plot latent space
data_pts = 1000
batch_size = 1
data_dir = 'data'
test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                    lambda x: x > 0, # binarisation de l'image
                    lambda x: x.float()]))

test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)
x = np.zeros((1, data_pts))
y = np.zeros((1, data_pts))
c = np.zeros((1, data_pts))
labels = ['0','1','2','3','4','5','6','7','8','9']

for batch_idx, (data, target) in enumerate(test_dataset):
    #mu,_ = vae.encoder(data.reshape((-1,784))) # Decomment if you use gaussian model
    mu = vae.encoder(data.reshape((-1,784))) # Decomment if you use bernoulli model
    x[0][batch_idx] = mu.detach().numpy()[0][0]
    y[0][batch_idx] = mu.detach().numpy()[0][1]
    c[0][batch_idx] = target.numpy()
    if batch_idx >= data_pts - 1:
        break
plt.figure()
plt.scatter(x,y,c=c)
cb = plt.colorbar()