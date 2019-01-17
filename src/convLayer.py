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
        
        self.pool1 = nn.MaxPool2d(convParams[0][5], return_indices=True)
        self.pool2 = nn.MaxPool2d(convParams[1][5], return_indices=True)
        self.pool3 = nn.MaxPool2d(convParams[2][5], return_indices=True)
        
    def forward(self,x):
        out1 = self.conv1(x)
        out1 = self.bn1(F.relu(out1))
#        pool_out1,pool_ind1 = self.pool1(out1)
        print('out1',out1.shape)
        out2 = self.conv2(out1)
        out2 = self.bn2(F.relu(out2))
        print('out2',out2.shape)
#        pool_out2,pool_ind2 = self.pool2(out2)
        out3 = self.conv3(out2)
        out3 = self.bn3(F.relu(out3))
        print('out3',out3.shape)
#        pool_out3,pool_ind3 = self.pool3(out3)
#        pool_indices = [(pool_ind1,out1.shape), (pool_ind2,out2.shape), (pool_ind3,out3.shape)]
        
        return out3 #pool_out3 , pool_indices
    
#    def getPoolIndices(self, i):
#        if i == 1:
#            return self.pool1
#        elif i == 2:
#            return self.pool2
#        elif i == 3:
#            return self.pool3

class deconvNN(nn.Module):
    def __init__(self, deconvParams):
        super(deconvNN,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(deconvParams[0][0], deconvParams[0][1], deconvParams[0][2], deconvParams[0][3], deconvParams[0][4])
        self.deconv2 = nn.ConvTranspose2d(deconvParams[1][0], deconvParams[1][1], deconvParams[1][2], deconvParams[1][3], deconvParams[1][4])
        self.deconv3 = nn.ConvTranspose2d(deconvParams[2][0], deconvParams[2][1], deconvParams[2][2], deconvParams[2][3], deconvParams[2][4])
        
        self.bn1 = nn.BatchNorm2d(deconvParams[0][1])
        self.bn2 = nn.BatchNorm2d(deconvParams[1][1])
        self.bn3 = nn.BatchNorm2d(deconvParams[2][1])
        
        self.unpool1 = nn.MaxUnpool2d(deconvParams[0][5])
        self.unpool2 = nn.MaxUnpool2d(deconvParams[1][5])
        self.unpool3 = nn.MaxUnpool2d(deconvParams[2][5])
        
        self.zeropad1 = nn.ZeroPad2d(deconvParams[0][6])
        self.zeropad2 = nn.ZeroPad2d(deconvParams[1][6])
        self.zeropad3 = nn.ZeroPad2d(deconvParams[2][6])
        
    def forward(self,x):
#        out1 = self.deconv1(x)
#        out1 = self.bn1(F.relu(out1))
#        print(out1.shape)
#        unpool_out1 = self.unpool1(out1,indices=pool_indices[2])
#        
#        out2 = self.deconv2(unpool_out1)
#        out2 = self.bn2(F.relu(out1))
#        unpool_out2 = self.unpool2(out2,indices=pool_indices[1])
#        
#        out3 = self.deconv2(unpool_out2)
#        out3 = F.relu(out3)
#        unpool_out3 = self.unpool3(out3,indices=pool_indices[0])
        
#        unpool_out1 = self.unpool1(x,indices=pool_indices[2][0], output_size=pool_indices[2][1])    
#        pad_out1 = self.zeropad1(unpool_out1)     
        out1 = self.deconv1(x)
        out1 = self.bn1(F.relu(out1))  
        print('out1',out1.shape)
#        unpool_out2 = self.unpool2(out1,indices=pool_indices[1][0], output_size=pool_indices[1][1])
#        pad_out2 = self.zeropad2(unpool_out2)
        out2 = self.deconv2(out1)
        out2 = self.bn2(F.relu(out2))
        print('out2',out2.shape)
#        unpool_out3 = self.unpool3(out2,indices=pool_indices[0][0], output_size=pool_indices[0][1]) 
        out3 = self.deconv3(out2)
        out3 = self.bn3(F.relu(out3))
        print('out3',out3.shape)
        
        return out3