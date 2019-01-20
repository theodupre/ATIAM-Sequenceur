# ATIAM-Sequenceur

Realization of a intelligent pattern sequencer using VAE in the context of the machine learning project at ATIAM Master.

# Getting Started

### Prerequisites
All the project is coded in python 3.7.0. Several libraries will me recommended to be installed : 
music21
sounddevice
torch
nsgt

## Generate pattern
To genereate new drum pattern, open the Activation_Matrix_To_Sound.py file and execute it. 
A window will appear and click in it to generate a sound. You can click multiple of time.
To finish the run, close this window. 

## Create activation matrices from midi
To create these activation matrices, open the Midi_To_Activation_Matrix.py file and execute it.
The example is done with the midi file HOUSE1.py

## Pattern Datasets
All the midi dataset is in the Dataset_Drum_Groove_Midi folder. 
The corresponding activation matrices are in the Dataset_Drum_Groove_Pattern folder.

## Audio Dataset and Reconstruction
We train the audio dataset on the gabor transform of signals.
To compute the gabor transform use : `python3 audio2nsgt.py -i dataset_audio/File/ -o results -d 1`
If the network works, to reconstruct the audio signal we use Inverse Gabor Transform from the library nsgt (Grill) whitch only give magnitude. Then, Griffin Lim iterative algorithm is used to reconstruct the phase. Our implementation is a modified version of bkvogel's. It does not work on some part of our audio dataset.

## Authors
Constance Douwes, 
Théophile Dupre,
Hadrien Marquez,
Robin Malzac,
Yann Teytaut


