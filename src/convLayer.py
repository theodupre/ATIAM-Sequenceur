#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:15:00 2019

@author: theophile
"""

import torch.nn as nn
import torch.nn.functional as F

class convNN(nn.Module):
    def __init__(self, convParams):
        super(convNN,self).__init__()
        self.conv1 = nn.Conv2d(convParams[0][0], convParams[0][1], convParams[0][2], convParams[0][3], convParams[0][4])
        self.conv2 = nn.Conv2d(convParams[1][0], convParams[1][1], convParams[1][2], convParams[1][3], convParams[1][4])
        self.conv3 = nn.Conv2d(convParams[2][0], convParams[2][1], convParams[2][2], convParams[2][3], convParams[2][4])
        
        self.bn1 = nn.BatchNorm2d(convParams[0][1])
        self.bn2 = nn.BatchNorm2d(convParams[1][1])
        self.bn3 = nn.BatchNorm2d(convParams[2][1])
        
    def forward(self,x):
        out1 = self.conv1(x)
        out1 = self.bn1(F.relu(out1))
        print('out1',out1.shape)
        out2 = self.conv2(out1)
        out2 = self.bn2(F.relu(out2))
        print('out2',out2.shape)
        out3 = self.conv3(out2)
        out3 = self.bn3(F.relu(out3))
        print('out3',out3.shape)     
        return out3
    

class deconvNN(nn.Module):
    def __init__(self, deconvParams):
        super(deconvNN,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(deconvParams[0][0], deconvParams[0][1], deconvParams[0][2], deconvParams[0][3], deconvParams[0][4])
        self.deconv2 = nn.ConvTranspose2d(deconvParams[1][0], deconvParams[1][1], deconvParams[1][2], deconvParams[1][3], deconvParams[1][4])
        self.deconv3 = nn.ConvTranspose2d(deconvParams[2][0], deconvParams[2][1], deconvParams[2][2], deconvParams[2][3], deconvParams[2][4])
        
        self.bn1 = nn.BatchNorm2d(deconvParams[0][1])
        self.bn2 = nn.BatchNorm2d(deconvParams[1][1])
        self.bn3 = nn.BatchNorm2d(deconvParams[2][1])

        self.zeropad1 = nn.ZeroPad2d(deconvParams[0][6])
        self.zeropad2 = nn.ZeroPad2d(deconvParams[1][6])
        self.zeropad3 = nn.ZeroPad2d(deconvParams[2][6])
        
    def forward(self,x):
   
        out1 = self.deconv1(x)
        out1 = self.bn1(F.relu(out1))  
        print('out1',out1.shape)
        
        pad_out1 = self.zeropad1(out1)
        print('pad_out1',pad_out1.shape)
        out2 = self.deconv2(pad_out1)
        out2 = self.bn2(F.relu(out2))
        print('out2',out2.shape)
        pad_out2 = self.zeropad2(out2)
        print('pad_out2',pad_out2.shape)
        out3 = self.deconv3(pad_out2)
        out3 = self.bn3(F.relu(out3))
        pad_out3 = self.zeropad3(out3)
        print('pad_out3',out3.shape)
        
        return pad_out3