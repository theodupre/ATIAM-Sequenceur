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
To genereate new drum pattern, open the `Activation_Matrix_To_Sound.py` file and execute it. 
A window will appear and click in it to generate a sound. You can click multiple of time.
To finish the run, close this window. 

## Create activation matrices from midi
To create these activation matrices, open the `Midi_To_Activation_Matrix.py` file and execute it.
The example is done with the midi file HOUSE1.py

## Pattern Datasets
All the midi dataset is in the Dataset_Drum_Groove_Midi folder. 
The corresponding activation matrices are in the Dataset_Drum_Groove_Pattern folder.

## Train Bernoulli variational autoencoder

`train_bernoulli.py` launches the training of the bernoulli variational autoencoder using the Pattern dataset. The VAE is said to be bernoulli because the ouput is optimized by the network to fit a bernoulli distribution as the input matrice ({0,1} values). 

## Train Gaussian variational autoencoder

`train_gaussian.py` launches the training of the gaussian variational autoencoder using the MNIST dataset. The VAE is said to be gaussian because the output is optimized by the network to fit a gaussian distribution.

## Audio Dataset and Reconstruction
We train the audio dataset on the gabor transform of signals.
To compute the gabor transform use : `python3 audio2nsgt.py -i dataset_audio/File/ -o results -d 1`
If the network works, to reconstruct the audio signal we use Inverse Gabor Transform from the library nsgt (Grill) whitch only give magnitude. Then, Griffin Lim iterative algorithm is used to reconstruct the phase. Our implementation is a modified version of bkvogel's. It does not work on some part of our audio dataset.


## Train Convolutional variational autoencoder

`train_audio.py` launches the training of the convolutional variation autoencoder using the audio dataset. The VAE is said to be convolutional because the hidden layers of the encoder and decoder are convolutional. The training doesn't work, the loss converges but the generation of a new sound ends up with matrices full of nan. More time would have been required to find the problem and to start a new training.

## Authors
Constance Douwes, 
Théophile Dupre,
Hadrien Marquez,
Robin Malzac,
Yann Teytaut


