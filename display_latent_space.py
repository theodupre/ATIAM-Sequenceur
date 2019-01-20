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
import pickle

from src import vae_gaussian as gaussian
from src import vae_bernoulli as bernoulli
from src import DatasetLoader as data
   
#%% Load vae, you can choose between gaussain and bernoulli models
D_in, D_enc, D_z, D_dec, D_out = 512, 800, 2, 800, 512 # Change D_z depending on dimensions of latent space
vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out); # change 'gaussian' to 'bernoulli' to change the model
vae.load_state_dict(torch.load('models/sequence/VAE_BERNOULLI_2_BETA_4_hid800')) # idem 
vae.eval()

#%% Sampling from latent space
sample = torch.randn(1,D_z)
print(sample)
sample = vae.decoder(sample)
y = sample.detach().numpy()
for i in range(y.size):
    if y[0][i] > 0.5:
        y[0][i] = 1
    else:
        y[0][i] = 0
x = y.reshape(8,64) 
plt.figure()
plt.imshow(x, cmap='gray',origin = 'lower')

#%% Plot 2D latent space
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
    mu,_ = vae.encoder(data.reshape((-1,D_in))) # Decomment if you use gaussian model
    #mu = vae.encoder(data.reshape((-1,784))) # Decomment if you use bernoulli model
    x[0][batch_idx] = mu.detach().numpy()[0][0]
    y[0][batch_idx] = mu.detach().numpy()[0][1]
    c[0][batch_idx] = target.numpy()
    if batch_idx >= data_pts - 1:
        break
plt.figure()
plt.scatter(x,y,c=c)
cb = plt.colorbar()

#%% Plot ND latent space in 2D using PCA
#data_pts = 1000
batch_size = 1
k = 2
data_dir = 'data/dataset_sequence/'
#test_dataset = datasets.MNIST(data_dir, train=False, download=True,
#                    transform=transforms.Compose([transforms.ToTensor(),
#                    lambda x: x > 0, # binarisation de l'image
#                    lambda x: x.float()]))
train_dataset = data.DatasetLoader(data_dir,transform=True)

#test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)

test_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
data_pts = len(train_dataset)
latent_space = np.zeros((D_z, data_pts));
label = np.zeros((1,data_pts))
for batch_idx, data in enumerate(test_loader):
    mu,_ = vae.encoder(data.reshape((-1,1024)))
    #mu = vae.encoder(data.reshape((-1,1024)))
    latent_space[:,batch_idx] = mu.detach().numpy()
    label[0,batch_idx] = 0 #target
    if batch_idx >= data_pts - 1:
        break
LS = torch.from_numpy(latent_space)
LS_mean = torch.mean(LS,0)
LS = LS - LS_mean.expand_as(LS)
U,S,V = torch.svd(LS)
C = torch.mm(torch.t(LS),U[:,:k])

plt.figure()
plt.scatter(C[:,0].numpy(),C[:,1].numpy(),c=label[0])
cb = plt.colorbar()

#%% PLot loss curves

saving_dir = 'models/sequence/'
pickle_in = open(saving_dir + 'VAE_BERNOULLI_2_BETA_4_hid800_loss.pickle',"rb")
loss_beta_5 = pickle.load(pickle_in)
#pickle_in = open(saving_dir + 'VAE_BERNOULLI_10_BETA_4_loss.pickle',"rb")
#loss_beta_10 = pickle.load(pickle_in)
#pickle_in = open(saving_dir + 'VAE_BERNOULLI_2_BETA_4_loss.pickle',"rb")
#loss_beta_2 = pickle.load(pickle_in)
#pickle_in = open(saving_dir + 'VAE_BERNOULLI_1_BETA_4_loss.pickle',"rb")
#loss_beta_1 = pickle.load(pickle_in)

x = np.linspace(1,200,200)
plt.plot(x,loss_beta_5['train_loss'],x,loss_beta_5['test_loss']) #,x,loss_beta_2['train_loss'],x,loss_beta_2['test_loss'],x,loss_beta_10['train_loss'],x,loss_beta_10['test_loss'],x,loss_beta_5['train_loss'],x,loss_beta_5['test_loss'])