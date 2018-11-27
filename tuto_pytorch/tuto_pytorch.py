#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:19:32 2018

@author: theophile
"""

from __future__ import print_function, division
import os
import torch
import numpy as np

dtype = torch.float
device = torch.device("cpu")

x = torch.tensor([[-1, -1],[-1,  1],[1, -1],[1,  1]], dtype = dtype) # Input patterns
y = np.array([0, 1, 1, 0])     

n_X = 2
n_Y = 1
n_hid = 2


# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 2, 2, 1


