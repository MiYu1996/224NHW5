#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """
    Class that have a skip connection controlled by a dynamic Gate
    """
    def __init__(self,word_emb):
        """
        Initialize the highway layer, where we have
        @param word_emb: Word Embedding size
        @x_conv is the 
        """
        super(Highway,self).__init__()
        self.word_emb = word_emb


        #### Default Values
        self.proj = None
        self.gate = None
        #### intialize two linear layers with bias
        self.proj = nn.Linear(self.word_emb,self.word_emb,bias = True)
        self.gate = nn.Linear(self.word_emb,self.word_emb,bias = True)

    def forward(self,x_conv):
        """
        This layer defines the sigmoid function
        @x_conv is the convolutional neural network
        output from the cnn
        """
        proj = F.relu(self.proj(x_conv))
        gate = torch.sigmoid(self.gate(x_conv))

        x = gate*proj + (1 - gate) * x_conv
        return x
        
        

### END YOUR CODE 

