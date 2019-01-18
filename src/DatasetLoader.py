#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:58:50 2019

@author: theophile
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

    def __len__(self):
        
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        
        if self.audio:
            file = os.listdir(self.root_dir)[idx]
            label = file[0]
            sample = (np.load(self.root_dir + os.listdir(self.root_dir)[idx]), label)
        else:
            sample = np.load(self.root_dir + os.listdir(self.root_dir)[idx])

        if self.transform:
            sample[0] = torch.from_numpy(sample[0]).float()
        else:
            sample = torch.from_numpy(sample).float()
        return sample
