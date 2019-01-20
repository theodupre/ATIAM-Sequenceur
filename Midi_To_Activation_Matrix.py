"""
Created on Mon Jan  7 17:43:20 2019

@author: Constance
"""

import matplotlib.pyplot as plt
from music21 import converter
import os
import numpy as np

# %% Function Definition

'''
Conversion of a midi note to a number correspondig the instrument used.
IN : Integer midi note
OUT : Integer correspondig the instrument used between 0 and 7.
'''


def get_instrument(midi):
    return {
        35: lambda: 0,  # Kick : 0
        36: lambda: 0,
        37: lambda: 1,  # Snare : 1
        38: lambda: 1,
        39: lambda: 2,  # Clap : 2
        40: lambda: 1,
        41: lambda: 5,  # Tom : 5
        42: lambda: 4,  # HHC : 4
        43: lambda: 5,
        44: lambda: 4,
        45: lambda: 5,
        46: lambda: 3,  # HHO : 3
        47: lambda: 5,
        48: lambda: 5,
        49: lambda: 6,  # Cymbales : 6
        50: lambda: 5,
        51: lambda: 6,
        52: lambda: 6,
        53: lambda: 6,
        55: lambda: 6,
        57: lambda: 6,
        59: lambda: 6,
        91: lambda: 1,
        93: lambda: 1
    }.get(midi, lambda: 7)()  # Perc : 7


'''
Creation of an activation matrix corresponding to the piece
IN : piece parser, nb_mesure the number of measure, quantification
OUT : activation_matrix array (8,quantification * nb_measure)
'''


def get_activation_matrix(piece, nb_measure, quantification):
    nb_instrument = 8       # fixed
    quantification = 64     # number of divison in one bar
    nb_measure = 1          # number of bars
    
    # initialisation of the activation matrix full of zeros
    activation_matrix = np.zeros((nb_instrument, quantification * nb_measure))

    for part in piece.parts:
        for this_note in part.recurse(classFilter=('Note', 'Chord')): 
            offset = this_note.offset
            metro = np.int(offset * quantification / 4)     # quantization of the offset of the note (bar 4/4)
            if metro < quantification * nb_measure:
                if this_note.isChord:        # if the note is a chord, we need to travel every note 
                    for note in this_note:
                        instrument = get_instrument(note.pitch.midi)
                        activation_matrix[instrument, metro] = 1
                else:                       # if the note is a single note
                    instrument = get_instrument(this_note.pitch.midi)
                    activation_matrix[instrument, metro] = 1

    return activation_matrix


'''
Import Midi function considering the midi file path
IN : path (string)
OUT : piece and activation_matrix
'''


def importMIDI(path):
    piece = converter.parse(path)
    activation_matrix = get_activation_matrix(piece, 1, 64)

    return piece, activation_matrix

# %% Main


root = r'Dataset_Drum_Groove_Midi'
list_dir = os.listdir(root)  # List of files in the directory

track_path = os.path.join(root, 'HOUSE1.mid')
piece, activation_matrix = importMIDI(track_path)

plt.figure(1, figsize=[15, 5])
plt.imshow(np.asarray(activation_matrix), origin='lower')
plt.set_cmap('gray')
#plt.savefig("test.eps", format="eps")

#Creation of the array dataset

# for data_file in list_dir:
#    if os.path.splitext(data_file)[1] == '.mid' or os.path.splitext(data_file)[1] == '.MID' : #Verification files in the floder are midi files
#
#            track_path = os.path.join(root,data_file)
#            piece,activation_matrix = importMIDI(track_path)
#
#            #Saving the activation_matrix numpy array
#            save_path = os.path.join('Dataset_Drum_Groove_Pattern',os.path.splitext(data_file)[0])
#            np.save(save_path,activation_matrix)
