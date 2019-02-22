#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    This is a class that runs the convolutional network
    """
    def __init__(self,echar,mword,eword,k=5):
        """
        Initialize the cnn dimensions
        @param echar the size for character embedding
        @param mword the maximum word length
        @param eword the size for the word embeding
        @param k the kernal size
        """
        super(CNN,self).__init__()
        self.echar = echar
        self.mword = mword
        self.eword = eword
        self.k = k
        ####initialize the convolutional layer
        self.conv = nn.Conv1d(self.echar,self.eword,self.k)
        self.Maxpool = nn.MaxPool1d(self.mword - self.k+1)

    def forward(self,x_reshape):
        x_conv = self.conv(x_reshape)
        x_conv_out = F.relu(x_conv)
        output = self.Maxpool(x_conv_out)
        return output
        
        
        

### END YOUR CODE

