#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Utility functions to train VAE Models
'''

import numpy as np
import matplotlib.pyplot as plt


def displayLoss(loss):
    
    epoch = range(len(loss['train_loss']))
    plt.plot(epoch, loss['train_loss'], epoch, loss['test_loss'])
    plt.show()