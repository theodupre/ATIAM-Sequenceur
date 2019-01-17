# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:53:37 2019

@author: dell
"""

from scipy.io import wavfile
import numpy as np
import os
import sounddevice as sd
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

from src import vae_bernoulli as bernoulli
 

#
fs, s_Kick = wavfile.read('DrumSound\Kick.wav')
fs, s_Snare = wavfile.read('DrumSound\Snare.wav')
fs, s_Clap = wavfile.read('DrumSound\Clap.wav')
fs, s_HHO = wavfile.read('DrumSound\HHO.wav')
fs, s_HHC = wavfile.read('DrumSound\HHC.wav')
fs, s_Tom = wavfile.read('DrumSound\Tom.wav')
fs, s_Cymb = wavfile.read('DrumSound\Cymb.wav')
fs, s_Percu = wavfile.read('DrumSound\Percu.wav')


#%%



   
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
    if y[0][i] > 0.2:
        y[0][i] = 1
    else:
        y[0][i] = 0
x = y.reshape(8,64) 
plt.figure()
plt.imshow(x, cmap='gray',origin = 'lower')




#root = r'C:\Users\dell\Documents\ATIAM\Info\ATIAM-Sequenceur\Dataset_Drum_Groove_Pattern'
#track_path = os.path.join(root,'(drums)_House2.npy')
#activation_matrix = np.load(track_path)
activation_matrix = x
#%%

bpm = 100
nb_instrument = 8       #fixed
quantification = 64     #number of divison in one measure
nb_measure = 1          #number of measures

lenght_sound_sec = 60/bpm * 4   #4 noires dans une mesure

lenght_sample = len(s_Kick)
lenght_sound_ech = int(lenght_sound_sec * fs) + lenght_sample

sound = np.zeros(lenght_sound_ech)
time_ech = 0

for activation in activation_matrix.T[:,:]:
    itemindex = np.where(activation==1)[0]
    print(itemindex)
    if itemindex.size == 0:
        time_ech = time_ech + int(lenght_sound_ech/quantification)
    else:
        for item in itemindex:
            if item == 0:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Kick
            if item == 1:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Snare
            if item == 2:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Clap
            if item == 3:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_HHO
            if item == 4:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_HHC
            if item == 5:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Tom
            if item == 6:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Cymb
            if item == 7:
                sound[time_ech:time_ech+lenght_sample] = sound[time_ech:time_ech+lenght_sample] + s_Percu
            
        time_ech = time_ech + int(lenght_sound_ech/quantification)
        print(time_ech)
        
        
sd.play(sound, fs)

#plt.figure(1, figsize=[15,5])
#plt.plot(sound)
#plt.savefig("sound.svg", format="svg")



