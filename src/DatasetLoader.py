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
        
#        self.conditioning = {'0': torch.tensor([1,0,0,0,0,0,0,0]),
#                      '1': torch.tensor([0,1,0,0,0,0,0,0]),
#                      '2': torch.tensor([0,0,1,0,0,0,0,0]),
#                      '3': torch.tensor([0,0,0,1,0,0,0,0]),
#                      '4': torch.tensor([0,0,0,0,1,0,0,0]),
#                      '5': torch.tensor([0,0,0,0,0,1,0,0]),
#                      '6': torch.tensor([0,0,0,0,0,0,1,0]),
#                      '7': torch.tensor([0,0,0,0,0,0,0,1])}
#        print(conditioning)


    def __len__(self):
        
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        
        if self.audio:
            conditioning = {'0': torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]),
                      '1': torch.tensor([0,1,0,0,0,0,0,0]).float(),
                      '2': torch.tensor([0,0,1,0,0,0,0,0]).float(),
                      '3': torch.tensor([0,0,0,1,0,0,0,0]).float(),
                      '4': torch.tensor([0,0,0,0,1,0,0,0]).float(),
                      '5': torch.tensor([0,0,0,0,0,1,0,0]).float(),
                      '6': torch.tensor([0,0,0,0,0,0,1,0]).float(),
                      '7': torch.tensor([0,0,0,0,0,0,0,1]).float()}
            file = os.listdir(self.root_dir)[idx]
            num = file[0]
            label = conditioning[num]
            data = np.abs(np.load(self.root_dir + os.listdir(self.root_dir)[idx])).transpose().reshape((1,410,157))
            data = torch.from_numpy(data).float()
            sample = (data, label)
        else:
            sample = np.load(self.root_dir + os.listdir(self.root_dir)[idx])

        if not self.audio:
            sample = torch.from_numpy(sample).float()
        return sample
