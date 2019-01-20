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
from src import vae_audio as audio
from src import DatasetLoader as data
   
#%% Load vae, you can choose between gaussain and bernoulli models
D_in, D_enc, D_z, D_dec, D_out = 512, 800, 2, 800, 512 # Change D_z depending on dimensions of latent space

#### Cyran's parameters of the death
# I'll make the code compatible with this fuckin shit, I'll send you a similar code in a while
conv1 = [1, 8, (11,5), (3,2), (5,2), (2,2)]
conv2 = [8, 16, (11,5), (3,2), (5,2), (2,2)]
conv3 = [16, 32, (11,5), (3,2), (5,2), (2,2)]
conv = [conv1, conv2, conv3]

#The MLP hidden Layers : [[in_dim,hlayer1_dim], [hlayer1_dim,hlayer2_dim], ...]
enc_h_dims = [[10240, 1024], [1024, 512]]
dec_h_dims = [[128, 1024], [1024, 10240]]
#The Deconv Layers: [in_channels, out_channels, kernel_size, stride, padding, output_padding]
deconv1 = [32, 16, (11,5), (3,2), (5,2), (2,2), (0,1,0,0)]
deconv2 = [16, 8, (11,5), (3,2), (5,2), (2,2), (0,0,1,0)]
deconv3 = [8, 1, (11,5), (3,2), (5,2), (2,2), (0,0,1,0)]
deconv = [deconv1, deconv2, deconv3]



N, input_size, D_z = 1, (410,157), 128

vae = audio.VAE_AUDIO(input_size, conv, enc_h_dims, D_z, deconv, dec_h_dims); # change 'gaussian' to 'bernoulli' to change the model
vae.load_state_dict(torch.load('models/audio/VAE_AUDIO_128_epoch20')) # idem 
vae.eval()

#%% Sampling from latent space
sample = torch.randn(1,D_z)
label = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.]).reshape(1,8)

sample = vae.decoder(sample, label)

y = sample[0].detach().numpy()

print(y.shape)
#for i in range(y.size):
#    if y[0][i] > 0.5:
#        y[0][i] = 1
#    else:
#        y[0][i] = 0
x = y.reshape(410,157) 
plt.figure()
plt.imshow(x,origin = 'lower')
plt.colorbar()

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

#%% PLot loss curves

saving_dir = 'models/audio/'
pickle_in = open(saving_dir + 'VAE_AUDIO_128_loss_epoch200.pickle',"rb")
loss_beta_2 = pickle.load(pickle_in)
#pickle_in = open(saving_dir + 'VAE_AUDIO_128_loss_epoch120.pickle',"rb")
#loss_beta_5 = pickle.load(pickle_in)
#pickle_in = open(saving_dir + 'VAE_AUDIO_128_loss_epoch100.pickle',"rb")
#loss_beta_1 = pickle.load(pickle_in)

x = np.linspace(1,201,201)
xx = np.linspace(1,111,111)
xxx = np.linspace(1,121,121)
plt.plot(x,loss_beta_2['train_loss'],x,loss_beta_2['test_loss'])#,xx,loss_beta_2['train_loss'],xx,loss_beta_2['test_loss'],xxx,loss_beta_5['train_loss'],xxx,loss_beta_5['test_loss'])
#plt.legend(('Train: dim = 1', 'Test : dim = 1','Train: dim = 2','Test : dim = 2','Train: dim = 5','Test : dim = 5'), loc = 'lower right')
#plt.xlabel('Epoch number')
#plt.ylabel('Loss')
#plt.title('Train and test losses depending on the latent space dimension')