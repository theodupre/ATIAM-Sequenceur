# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:53:37 2019

@author: dell
"""

from scipy.io import wavfile
import numpy as np
import os
import sounddevice as sd
import time
import matplotlib.pyplot as plt

 
# Wait for 300 milliseconds
# .3 can also be used

fs, s_Kick = wavfile.read('DrumSound\Kick.wav')
fs, s_Snare = wavfile.read('DrumSound\Snare.wav')
fs, s_Clap = wavfile.read('DrumSound\Clap.wav')
fs, s_HHO = wavfile.read('DrumSound\HHO.wav')
fs, s_HHC = wavfile.read('DrumSound\HHC.wav')
fs, s_Tom = wavfile.read('DrumSound\Tom.wav')
fs, s_Cymb = wavfile.read('DrumSound\Cymb.wav')
fs, s_Percu = wavfile.read('DrumSound\Percu.wav')

plt.plot(s_HHO)

#%% Main

root = r'C:\Users\dell\Documents\ATIAM\Info\ATIAM-Sequenceur\Dataset_Drum_Groove_Pattern'
track_path = os.path.join(root,'HOUSE1.npy')
activation_matrix = np.load(track_path)


#for activation in activation_matrix.T[:,:]:
#    itemindex = np.where(activation==1)[0]
#    if itemindex.size == 0:
#        sound = np.zeros(len(s_Kick))
#        time.sleep(.020)
#    else:
#        print(itemindex)
#        sound = np.zeros(len(s_Kick))
#        for item in itemindex:
#            
#            if item == 0:
#                sound = sound + s_Kick
#                sd.stream(s_Kick, fs)
#            if item == 1:
#                sound = sound + s_Snare
#                #sd.play(s_Snare, fs)
#            if item == 2:
#                sound = sound + s_Clap
#                #sd.play(s_Clap, fs)
#            if item == 3:
#                sound = sound + s_HHO
#                #sd.play(s_HHO, fs)
#            if item == 4:
#                sound = sound + s_HHC
#                #sd.play(s_HHC, fs)
#            if item == 5:
#                sound = sound + s_Tom
#                #sd.play(s_Tom, fs)
#            if item == 6:
#                sound = sound + s_Cymb
#                #sd.play(s_Cymb, fs)
#            if item == 7:
#                sound = sound + s_Percu
#                #sd.play(s_Percu, fs)
#        time.sleep(.020)
#        sd.play(sound, fs)

duration = 1  # seconds



def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

with sd.RawStream(channels = 2, dtype = 'int24', callback = s_Kick):
    sd.sleep(int(duration * 1000))

