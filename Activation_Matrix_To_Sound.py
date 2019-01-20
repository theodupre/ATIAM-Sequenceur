# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:53:37 2019

@author: Constance
"""

from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import torch

from src import vae_bernoulli as bernoulli
from tkinter import *

# Load wavefile samples, to be remplaced with generated sounds
fs, s_Kick = wavfile.read(r'DrumSound\Kick.wav')
fs, s_Snare = wavfile.read(r'DrumSound\Snare.wav')
fs, s_Clap = wavfile.read(r'DrumSound\Clap.wav')
fs, s_HHO = wavfile.read(r'DrumSound\HHO.wav')
fs, s_HHC = wavfile.read(r'DrumSound\HHC.wav')
fs, s_Tom = wavfile.read(r'DrumSound\Tom.wav')
fs, s_Cymb = wavfile.read(r'DrumSound\Cymb.wav')
fs, s_Percu = wavfile.read(r'DrumSound\Percu.wav')

# %% Load vae, you can choose between gaussain and bernoulli models
# Change D_z depending on dimensions of latent space
D_in, D_enc, D_z, D_dec, D_out = 512, 800, 2, 800, 512
# change 'gaussian' to 'bernoulli' to change the model
vae = bernoulli.VAE_BERNOULLI(D_in, D_enc, D_z, D_dec, D_out)
vae.load_state_dict(torch.load(
    'models/sequence/VAE_BERNOULLI_2_BETA_4_hid800'))  # idem
vae.eval()


# %% Sampling from latent space
'''
Generation of an activation matrix depending of the sample chosen
IN : sample (tensor with shape 1xD_z)
OUT : activation_matrix (8x64)
'''
def generate_matrix(sample):
    print(sample)
    sample = vae.decoder(sample)
    y = sample.detach().numpy()
    for i in range(y.size):
        if y[0][i] > 0.3: #binarization of the matrix over a threshold
            y[0][i] = 1
        else:
            y[0][i] = 0
    activation_matrix = y.reshape(8, 64)
    
#    plt.figure(1)
#    plt.imshow(activation_matrix, cmap='gray',origin = 'lower')
#    plt.savefig("matrice.eps", format="eps")

    return activation_matrix

'''
Play the sound corresponding to an activation matrix
IN : activation_matrix (8x64)
'''  
def play_sound(activation_matrix):
    
    bpm = 120
    quantification = 64         # number of divison in one measure

    lenght_sound_sec = 60 / bpm * 4  # 4/4 measure

    lenght_sample = len(s_Kick)
    lenght_sound_ech = int(lenght_sound_sec * fs) + lenght_sample

    sound = np.zeros(lenght_sound_ech)
    time_ech = 0

    for activation in activation_matrix.T[:, :]:
        itemindex = np.where(activation == 1)[0]
        if time_ech + lenght_sample < len(sound):
            if itemindex.size == 0:
                time_ech = time_ech + int(lenght_sound_ech / quantification)
            else:

                for item in itemindex:
                    if item == 0:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Kick
                    if item == 1:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Snare
                    if item == 2:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Clap
                    if item == 3:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_HHO
                    if item == 4:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_HHC
                    if item == 5:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Tom
                    if item == 6:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Cymb
                    if item == 7:
                        sound[time_ech:time_ech +
                              lenght_sample] = sound[time_ech:time_ech + lenght_sample] + s_Percu

                time_ech = time_ech + \
                    int((lenght_sound_ech + 1) / quantification)

    sd.play(sound, fs)
    
#    plt.figure(2)
#    plt.plot(sound)
#    plt.savefig("sound.eps", format="eps")


'''
Callback function that is the answer from an event and create a tensor from
x and y coordonates of the window with shape 400 x 400. 
IN : event coming from the <Button-1> wich corresponds to the left mouse click
'''

def callback(event):
    frame.focus_set()
    python_green = "#476042"

    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)

    frame.create_oval(x1, y1, x2, y2, fill=python_green)
    sample = torch.tensor([[(event.x - 200) / 50, (event.y - 200) / 50]]) #samples are form -4 to 4
    play_sound(generate_matrix(sample))


# %%
root = Tk()

frame = Canvas(root, width=400, height=400, bg='ivory')
frame.bind("<Button-1>", callback)
frame.pack()

root.mainloop()
