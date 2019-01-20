#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:58:50 2019

@author: theophile
"""

"""
Class to load specific dataset. 

One dataset is numpy arrays (128,8) representing activation 
matrices (contained in 'Dataset_Drum_Groove_Pattern/')
The other is numpy arrays representing (410,157) gabor transform 
of drum elements (contained in 'dataset_audio/')

@param: root_dir: directory where the data should be loaded from
		tranform: is the data need to be transformed (default: None)
		audio: is the data is gabor transform files (default: False)

"""

import os
import numpy as np
import torch
from  torch.utils.data import Dataset

class DatasetLoader(Dataset):

    def __init__(self, root_dir, transform=None, audio=False):

        self.root_dir = root_dir
        self.transform = transform
        self.audio = audio

	# Return number of files in @root_dir (i.e. number of data in the dataset)
    def __len__(self):
        
        return len(os.listdir(self.root_dir))
	
	# Get one item from the dataset
    def __getitem__(self, idx):
        
		# if this datset is audio, the type of audio (i.e. kick, snare, etc) is required
		# the type is coded as the first char of the file name
        if self.audio:
			
			# dict with correspondance between labels in file_name and conditionning vectors used in the network
            conditioning = {'0': torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]),
                      '1': torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.]),
                      '2': torch.tensor([0.,0.,1.,0.,0.,0.,0.,0.]),
                      '3': torch.tensor([0.,0.,0.,1.,0.,0.,0.,0.]),
                      '4': torch.tensor([0.,0.,0.,0.,1.,0.,0.,0.]),
                      '5': torch.tensor([0.,0.,0.,0.,0.,1.,0.,0.]),
                      '6': torch.tensor([0.,0.,0.,0.,0.,0.,1.,0.]),
                      '7': torch.tensor([0.,0.,0.,0.,0.,0.,0.,1.])}
            file = os.listdir(self.root_dir)[idx]
            num = file[0] # Get the coded type
            label = conditioning[num] 
			# Get the absolute value of the gabor transform and reshape the data
            data = np.abs(np.load(self.root_dir + os.listdir(self.root_dir)[idx])).transpose().reshape((1,410,157))
            data = torch.from_numpy(data).float()
            sample = (data, label)
        else:
            sample = np.load(self.root_dir + os.listdir(self.root_dir)[idx])

        if not self.audio:
            sample = torch.from_numpy(sample).float()
        return sample
