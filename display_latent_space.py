#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:56:32 2018

@author: theophile

Scripts containing several parts displaying results

"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import pickle

from src import vae_gaussian as gaussian
from src import vae_bernoulli as bernoulli
from src import vae_audio as audio
from src import DatasetLoader as data
   
#%% Load vae, you can choose between gaussain and bernoulli models
D_in, D_enc, D_z, D_dec, D_out = 512, 800, 2, 800, 512 

vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_enc, D_out); 
vae.load_state_dict(torch.load('models/sequence/VAE_BERNOULLI_2_BETA_4_hid800')) #
vae.eval()

#%% Sampling from latent space
sample = torch.randn(1,D_z)
sample = vae.decoder(sample)
y = sample.detach().numpy()
for i in range(y.size):
    if y[0][i] > 0.2:
        y[0][i] = 1
    else:
        y[0][i] = 0
x = y.reshape(8,64) 
plt.figure()
plt.imshow(x,origin = 'lower')
plt.show()

#%% Plot ND latent space in 2D using PCA
#data_pts = 1000
batch_size = 1
k = 2
data_dir = 'data/dataset_sequence/'
#train_dataset = datasets.MNIST(data_dir, train=False, download=True,
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
    mu,_ = vae.encoder(data.reshape((-1,512)))

    latent_space[:,batch_idx] = mu.detach().numpy()
    label[0,batch_idx] = 0
    if batch_idx >= data_pts - 1:
        break
print(latent_space.shape)
LS = torch.from_numpy(latent_space)
LS_mean = torch.mean(LS,0)
LS = LS - LS_mean.expand_as(LS)
U,S,V = torch.svd(LS)
C = torch.mm(torch.t(LS),U[:,:k])
#%%
fig = plt.figure()
#plt.scatter(C[:,0].numpy(),C[:,1].numpy(),c=label[0])
plt.scatter(latent_space[0,:],latent_space[1,:],c=label[0])
plt.suptitle('Latent space representation of test sequence data (mean only)')
plt.xlabel('1st dim of mean')
plt.ylabel('2nd dim of mean')
#cb = plt.colorbar()
plt.show()

#%% PLot loss curves

saving_dir = 'models/audio/'
pickle_in = open(saving_dir + 'VAE_AUDIO_128_loss_epoch200.pickle',"rb")
loss_beta_2 = pickle.load(pickle_in)
x = np.linspace(1,201,201)
plt.plot(x,loss_beta_2['train_loss'],x,loss_beta_2['test_loss'])
plt.legend(('Train loss', 'Test loss'))
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.title('Train and test losses for the audio vae (doesn\'t work)')
plt.show()